"""第83バッチ: θ 較正パイロット + 思考頻度の観測装置化のテスト。

正典: docs/plans/source/cognition-design-record.md §2.6(共通スケール較正)・§3.5
      (パイロットで発火数分布と総推論量を実測)・§8(観測量)/
      docs/plans/source/physics-instructions.md Part P5。

守るもの(検収基準の順)
  (1) calibrate_theta が**決定論**(同じ引数で 2 回走らせてバイト一致)
  (2) 較正が触るのは `cognition.fire.theta_scale` **1 本だけ**
      = 較正テーブル(文脈別 θ 水準)にも conf にも書き戻さない(設計 §2.6「個体差は残す」)
  (3) analyze_firing の 5 解析が**合成入力に対して期待どおりの向き**に出る
      (バースト検出・バースト性・間隔分布・連鎖規則・予測誤差の復元)
  (4) 発火連鎖は **lag=1 step の規則**でしか繋がない(観測凍結の構造から一意)
  (5) mock ラン end-to-end で 5 解析が完走し、出力が決定論
  (6) 可変思考 ON/OFF の**両方**で既存指標(エコー / 伝播 / 規範 stage)の解析が完走する
      = P5「可変思考 ON/OFF の対照ラン」の**配線確認**(数値の比較主張はしない)

house style: scripts/ を path 追加して import(tests/test_analyze_specialization.py に倣う)。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_firing as AF                            # noqa: E402
import calibrate_theta as CT                           # noqa: E402

from society.cognition import calib as CALIB           # noqa: E402
from society.config import load_config                 # noqa: E402
from society.engine.simulation import Simulation       # noqa: E402

FIRE_ON = {
    "cognition.fire.enabled": "true",
    "cognition.fire.theta_scale": "0.03125",   # data/calib/theta_scale.json の較正値
    "cognition.fire.max_contrib": "8",         # usable(7)以上 = contrib が完全になる
    "cognition.g_update.enabled": "true",
    "cognition.g_update.log_every_steps": "1",
}


def _run(tmp_path: Path, name: str, *, n_steps: int = 48, n_agents: int = 20,
         **ov) -> Path:
    out = tmp_path / name
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    Simulation(load_config(dot), out_dir=out).run()
    return out


# --------------------------------------------------------------------------- #
# (3) 統計の単体: バースト性(Goh & Barabási 2008 / Kim & Jo 2016)
# --------------------------------------------------------------------------- #
def test_burstiness_is_minus_one_for_a_perfectly_regular_series():
    out = AF.burstiness([10.0] * 50)
    assert out["cv"] == 0.0
    assert out["B"] == pytest.approx(-1.0)


def test_burstiness_rises_when_intervals_become_heterogeneous():
    regular = AF.burstiness([10.0] * 40)
    bursty = AF.burstiness([1.0] * 38 + [500.0, 500.0])
    assert bursty["B"] > regular["B"]
    assert bursty["B"] > 0.0                    # ポアソンより不均質
    # 有限長補正は素の B と別値であり、必ず併記される
    assert bursty["A_n"] is not None and bursty["A_n"] != bursty["B"]


def test_burstiness_is_undefined_for_degenerate_input():
    assert AF.burstiness([])["B"] is None
    assert AF.burstiness([5.0])["B"] is None


# --------------------------------------------------------------------------- #
# (3) Kleinberg 2 状態バースト検出
# --------------------------------------------------------------------------- #
def test_kleinberg_detects_an_injected_burst_window():
    counts = [1] * 30 + [18] * 8 + [1] * 30
    at_risk = [40] * len(counts)
    out = AF.kleinberg_bursts(counts, at_risk, s=2.0, gamma=1.0)
    assert out["status"] == "ok"
    assert out["bursts"], "注入したバーストが検出されない"
    top = max(out["bursts"], key=lambda b: b["weight"])
    # 注入区間 [30, 37] を覆っている
    assert top["start_bin"] <= 30 and top["end_bin"] >= 37


def test_kleinberg_finds_nothing_in_a_flat_series():
    counts = [4] * 60
    out = AF.kleinberg_bursts(counts, [40] * 60, s=2.0, gamma=1.0)
    assert out["status"] == "ok"
    assert out["bursts"] == []


def test_kleinberg_reports_status_instead_of_crashing_on_empty_input():
    assert AF.kleinberg_bursts([], [])["status"] == "too_few_bins"
    assert AF.kleinberg_bursts([0, 0, 0], [5, 5, 5])["status"] == "no_events"


# --------------------------------------------------------------------------- #
# (3) ① 思考間隔分布
# --------------------------------------------------------------------------- #
def _cog(step, aid, ctx="walking", kind="cog_fire", reason="periodic", **p):
    return {"step": step, "sim_min": step * 10, "agent_id": aid, "kind": kind,
            "x": 0.0, "y": 0.0, "p": {"ctx": ctx, "reason": reason, **p}}


def test_intervals_are_per_agent_and_per_context():
    cog = ([_cog(i, 1, "walking") for i in range(0, 6)]
           + [_cog(i, 2, "resting") for i in range(0, 12, 3)])
    res = AF.analyze_intervals(cog)
    assert res["overall"]["n"] == 5 + 3
    assert res["by_context"]["walking"]["mean"] == pytest.approx(10.0)
    assert res["by_context"]["resting"]["mean"] == pytest.approx(30.0)
    assert res["by_agent"]["n_agents"] == 2
    # ヒストグラムの総数は間隔の総数に一致する(取りこぼしゼロ)
    assert sum(b["n"] for b in res["histogram_min"]) == res["overall"]["n"]


def test_intervals_ignore_zero_length_gaps():
    """同一 step に social イベントと期限イベントが並んでも 0 分の間隔は数えない。"""
    cog = [_cog(0, 1), _cog(0, 1, kind="cog_event"), _cog(1, 1)]
    assert AF.analyze_intervals(cog)["overall"]["n"] == 1


# --------------------------------------------------------------------------- #
# (3)(4) ④ 発火連鎖グラフ
# --------------------------------------------------------------------------- #
def _fire(step, aid, contrib, s=5.0, x=0.0, y=0.0):
    return {"step": step, "sim_min": step * 10, "agent_id": aid, "kind": "cog_fire",
            "x": x, "y": y,
            "p": {"reason": "salience", "ctx": "walking", "s": s, "theta": 1.0,
                  "contrib": contrib}}


def _cause(step, kind, aid, payload, x=0.0, y=0.0):
    return {"step": step, "sim_min": step * 10, "agent_id": aid, "kind": kind,
            "x": x, "y": y, "p": payload}


def test_chain_recovers_a_named_hear_path_with_high_confidence():
    """A が話し B が聞いた(L1 hear が両者を名指す)→ 確度 high の A→B。"""
    cog = [_fire(10, 2, [["ext.heard", 3.0]])]
    causes = {(9, "hear"): [_cause(9, "hear", 2, {"speaker": 7})]}
    res = AF.analyze_chains(cog, causes, lag=1, radius_m=40.0, min_contrib=0.5,
                            max_examples=5)
    assert res["n_candidate_edges"] == 1
    edge = res["edges"][0]
    assert (edge["from"], edge["to"], edge["confidence"]) == (7, 2, "high")
    assert edge["channel"] == "ext.heard" and edge["via"] == "hear"
    assert edge["cause_step"] == 9


def test_chain_recovers_a_named_dm_path_from_the_sender_side():
    cog = [_fire(10, 2, [["ext.heard", 3.0]])]
    causes = {(9, "dm"): [_cause(9, "dm", 5, {"to": 2})]}
    res = AF.analyze_chains(cog, causes, lag=1, radius_m=40.0, min_contrib=0.5,
                            max_examples=5)
    assert [(e["from"], e["to"], e["confidence"]) for e in res["edges"]] \
        == [(5, 2, "high")]


def test_chain_marks_nearby_paths_medium_when_unique_and_low_when_ambiguous():
    cog = [_fire(10, 2, [["ext.crowd_local", 3.0]], x=0.0, y=0.0)]
    one = {(9, "arrive"): [_cause(9, "arrive", 8, {"node": "n1"}, x=5.0, y=0.0)]}
    res1 = AF.analyze_chains(cog, one, lag=1, radius_m=40.0, min_contrib=0.5,
                             max_examples=5)
    assert [e["confidence"] for e in res1["edges"]] == ["medium"]

    two = {(9, "arrive"): [_cause(9, "arrive", 8, {}, x=5.0, y=0.0),
                           _cause(9, "arrive", 9, {}, x=6.0, y=0.0)]}
    res2 = AF.analyze_chains(cog, two, lag=1, radius_m=40.0, min_contrib=0.5,
                             max_examples=5)
    assert sorted(e["confidence"] for e in res2["edges"]) == ["low", "low"]
    assert all(e["n_candidates"] == 2 for e in res2["edges"])


def test_chain_respects_the_radius_and_the_one_step_lag():
    """観測凍結の構造から因果ラグは 1 step。ずれた step / 遠い相手は繋がない。"""
    cog = [_fire(10, 2, [["ext.crowd_local", 3.0]], x=0.0, y=0.0)]
    far = {(9, "arrive"): [_cause(9, "arrive", 8, {}, x=500.0, y=0.0)]}
    assert AF.analyze_chains(cog, far, lag=1, radius_m=40.0, min_contrib=0.5,
                             max_examples=5)["n_candidate_edges"] == 0
    wrong_step = {(6, "arrive"): [_cause(6, "arrive", 8, {}, x=1.0, y=0.0)]}
    assert AF.analyze_chains(cog, wrong_step, lag=1, radius_m=40.0,
                             min_contrib=0.5, max_examples=5)["n_candidate_edges"] == 0


def test_chain_does_not_invent_paths_for_body_channels():
    """身体チャンネル由来の発火は他者経路を持たない(自己由来として数える)。"""
    cog = [_fire(10, 2, [["body.drive", 3.0]])]
    causes = {(9, "hear"): [_cause(9, "hear", 2, {"speaker": 7})]}
    res = AF.analyze_chains(cog, causes, lag=1, radius_m=40.0, min_contrib=0.5,
                            max_examples=5)
    assert res["n_candidate_edges"] == 0
    assert res["n_fire_self_origin"] == 1


def test_chain_drops_contributions_below_the_threshold():
    cog = [_fire(10, 2, [["ext.heard", 0.2]])]
    causes = {(9, "hear"): [_cause(9, "hear", 2, {"speaker": 7})]}
    res = AF.analyze_chains(cog, causes, lag=1, radius_m=40.0, min_contrib=0.5,
                            max_examples=5)
    assert res["n_candidate_edges"] == 0


# --------------------------------------------------------------------------- #
# (3) ⑤ 予測誤差の復元(contrib ÷ g)
# --------------------------------------------------------------------------- #
def test_prediction_divides_contrib_by_the_previous_step_g():
    cog = [_fire(10, 2, [["ext.heard", 4.0]])]
    meta = {"max_contrib": 8, "usable_channels": ["ext.heard"], "g_update": True}
    g_table = {(9, 2): {"ext.heard": 2.0}, (10, 2): {"ext.heard": 8.0}}
    res = AF.analyze_prediction(cog, [], meta, g_table, hit_sigma=1.0)
    # 発火判定が読む g は **1 step 前** の値 = 2.0 → 誤差 4.0/2.0 = 2.0σ
    assert res["by_channel"]["ext.heard"]["mean_err_sigma"] == pytest.approx(2.0)
    assert res["by_channel"]["ext.heard"]["hit_rate"] == 0.0
    assert res["err_recoverable"] is True


def test_prediction_declares_when_g_cannot_be_recovered():
    cog = [_fire(10, 2, [["ext.heard", 4.0]])]
    meta = {"max_contrib": 8, "usable_channels": ["ext.heard"], "g_update": True}
    res = AF.analyze_prediction(cog, [], meta, {}, hit_sigma=1.0)
    assert res["err_recoverable"] is False
    assert "unavailable" in res["g_source"]


def test_prediction_counts_unlisted_channels_as_perfect_only_when_complete():
    cog = [_fire(10, 2, [["ext.heard", 4.0]])]
    full = {"max_contrib": 8, "usable_channels": ["ext.heard", "ext.encounter"],
            "g_update": False}
    res = AF.analyze_prediction(cog, [], full, {}, hit_sigma=1.0)
    assert res["contrib_complete"] is True
    assert res["by_channel"]["ext.encounter"]["hit_rate"] == 1.0   # 誤差 0 = 的中

    cut = {"max_contrib": 1, "usable_channels": ["ext.heard", "ext.encounter"],
           "g_update": False}
    res2 = AF.analyze_prediction(cog, [], cut, {}, hit_sigma=1.0)
    assert res2["contrib_complete"] is False
    assert "ext.encounter" not in res2["by_channel"]               # 0 で埋めない


# --------------------------------------------------------------------------- #
# (5) end-to-end: mock ランで 5 解析が完走し、出力が決定論
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fire_run(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("b83")
    return _run(tmp, "fire_on", n_steps=48, n_agents=20,
                **{**FIRE_ON, "cognition.watch.enabled": "true"})


def _analyze(run_dir: Path, out: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze_firing.py"),
         str(run_dir), "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"analyze_firing が失敗: {proc.stderr[-2000:]}"
    return json.loads((out / "firing_report.json").read_text(encoding="utf-8"))


def test_analyze_firing_runs_all_five_analyses(fire_run, tmp_path):
    res = _analyze(fire_run, tmp_path / "a")
    for key in ("intervals", "salience_sources", "timeline", "chains", "prediction"):
        assert key in res, f"{key} が出ていない"
    assert res["intervals"]["overall"]["n"] > 0
    assert res["timeline"]["status"] == "ok"
    assert res["salience_sources"]["n_salience"] > 0
    assert res["prediction"]["by_channel"], "チャンネル別の予測誤差が空"
    # max_contrib(8) >= usable(7)なので寄与内訳は完全であると自己申告できる
    assert res["salience_sources"]["contrib_complete"] is True
    # 物理層は未実装であることを**黙って省略せず**宣言する
    assert "not_implemented" in res["physics_zones"]
    assert res["limits"], "限界の併記が無い"


def test_analyze_firing_is_deterministic(fire_run, tmp_path):
    _analyze(fire_run, tmp_path / "a")
    _analyze(fire_run, tmp_path / "b")
    for name in ("firing_report.json", "firing_report.md"):
        assert (tmp_path / "a" / name).read_bytes() == \
               (tmp_path / "b" / name).read_bytes(), f"{name} が非決定"


def test_analyze_firing_finds_real_chains_in_a_mock_run(fire_run, tmp_path):
    res = _analyze(fire_run, tmp_path / "a")
    ch = res["chains"]
    assert ch["n_candidate_edges"] > 0, "発火連鎖の候補が 1 本も出ない"
    assert ch["by_confidence"]["high"] > 0, "名指し由来の確度 high が出ない"
    for edge in ch["edges"]:
        assert edge["cause_step"] == edge["step"] - 1        # lag の一貫性
        assert edge["from"] != edge["to"]                    # 自己ループを作らない
    assert "因果の証明ではない" in ch["caveat"]


def test_analyze_firing_refuses_a_run_without_cognitive_events(tmp_path):
    plain = _run(tmp_path, "no_fire", n_steps=12, n_agents=6)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze_firing.py"), str(plain)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode != 0
    assert "cognition.fire" in (proc.stderr + proc.stdout)


# --------------------------------------------------------------------------- #
# (1)(2) θ 較正パイロット
# --------------------------------------------------------------------------- #
def _calibrate(tmp_path: Path, tag: str, out_name: str = "theta.json") -> Path:
    out = tmp_path / out_name
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "calibrate_theta.py"),
         "--agents", "12", "--steps", "24", "--span", "0", "--max-iter", "0",
         "--max-extend", "0", "--no-scale-check", "--target", "1.0",
         "--run-dir", str(tmp_path / tag), "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"calibrate_theta が失敗: {proc.stderr[-3000:]}"
    return out


def test_calibrate_theta_is_deterministic(tmp_path):
    a = _calibrate(tmp_path, "ra", "a.json")
    b = _calibrate(tmp_path, "rb", "b.json")
    assert a.read_bytes() == b.read_bytes(), "calibrate_theta が非決定"


def test_calibrate_theta_freezes_scale_estimate_and_provenance(tmp_path):
    doc = json.loads(_calibrate(tmp_path, "rc").read_text(encoding="utf-8"))
    assert doc["meta"]["kind"] == "theta_scale"
    assert doc["meta"]["payload_sha256"] == CALIB.payload_sha256(doc)
    # 設計 §2.6: 触るのは全体スケールだけ / 個体差は残す
    assert "theta_scale" in doc["meta"]["policy"]["lever"]
    assert "個体差" in doc["meta"]["policy"]["preserved"]
    # §3.5: 総推論量の見積が出ている
    rows = doc["inference_estimate"]["rows"]
    assert [r["n_agents"] for r in rows] == [100, 1000, 10000]
    assert all(r["llm_calls"] > 0 for r in rows)
    # 適用は明示上書き(conf へは書き戻さない)
    assert doc["result"]["apply_override"].startswith("cognition.fire.theta_scale=")


def test_calibrate_theta_does_not_write_back_into_the_conf_table(tmp_path):
    """★較正結果を conf/cognition/calib_default.yaml に書き戻さない(設計判断)。"""
    path = REPO_ROOT / CALIB.CALIB_DEFAULT_REL
    before = CALIB.file_sha256(path)
    doc = json.loads(_calibrate(tmp_path, "rd").read_text(encoding="utf-8"))
    assert CALIB.file_sha256(path) == before, "較正テーブルが書き換えられた"
    # 凍結ファイルは「どのテーブルで測ったか」を sha256 で指す
    assert doc["meta"]["inputs"]["calib_table"]["sha256"] == before
    # テーブル自身の provisional 宣言は維持される
    assert CALIB.load_calib(None)["table"]["status"] == "provisional"


def test_calibrate_theta_pilot_only_sweeps_the_global_scale(monkeypatch, tmp_path):
    """パイロットが渡す dotlist が θ の**全体スケール以外**を触らないことを固定する。"""
    seen: list[list[str]] = []

    class _Sim:
        def __init__(self, cfg, out_dir=None):
            seen.append(list(cfg._dotlist_probe))

        def run(self):
            raise RuntimeError("stop")

    def _load(dot):
        cfg = load_config(dot)
        cfg._dotlist_probe = dot
        return cfg

    monkeypatch.setattr(CT, "load_config", _load)
    monkeypatch.setattr(CT, "Simulation", _Sim)
    with pytest.raises(RuntimeError):
        CT.pilot(0.5, agents=4, steps=6, seed=1, run_dir=tmp_path,
                 extra=[], watch=False, tag="p")
    dot = seen[0]
    assert "cognition.fire.theta_scale=0.5" in dot
    # 文脈別 θ 水準・個体倍率・恒常性の目標には触れない
    assert not any(k.startswith("cognition.fire.period") for k in dot)
    assert not any("salience." in k for k in dot)
    assert not any("theta_target_per_day" in k for k in dot)
    # 恒常性だけは較正中に切る(開ループ利得を測るため)= その宣言も固定する
    assert "cognition.g_update.theta_mu=0.0" in dot


def test_frozen_theta_scale_file_is_consistent_if_present():
    """リポジトリに凍結済みの theta_scale.json があれば自己整合であること。"""
    path = REPO_ROOT / "data" / "calib" / "theta_scale.json"
    if not path.exists():
        pytest.skip("theta_scale.json 未凍結")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["meta"]["payload_sha256"] == CALIB.payload_sha256(doc)
    assert doc["meta"]["inputs"]["calib_table"]["sha256"] == \
        CALIB.file_sha256(REPO_ROOT / CALIB.CALIB_DEFAULT_REL), \
        "較正テーブルが変わっている: scripts/calibrate_theta.py で測り直すこと"
    res = doc["result"]
    assert res["rel_error"] <= res["tol"], "凍結値が許容誤差に入っていない"


# --------------------------------------------------------------------------- #
# (6) 可変思考 ON/OFF の対照 = **配線確認だけ**(数値の比較主張はしない)
# --------------------------------------------------------------------------- #
ECHO_COLS = ("echo_max", "echo_utterance_rate")
# 伝播 KPI(第70 でエコー除外の新列が並記された側)
PROPAGATION_COLS = ("transmission_novel", "transmission_novel_rate", "total_adoptions")


@pytest.mark.parametrize("fire", [False, True])
def test_existing_metrics_analyses_run_under_both_fire_settings(tmp_path, fire):
    """P5「可変思考 ON/OFF の対照ラン」の器: 既存指標の解析が**両方で完走**する。

    ★ここで比べるのは「解析が通るか」だけで、**数値の大小は一切主張しない**
      (対照の主張は seed 複数本と事前登録の領分)。
    """
    ov = dict(FIRE_ON) if fire else {}
    out = _run(tmp_path, f"ctl_{int(fire)}", n_steps=48, n_agents=20, **ov)

    l2 = pq.read_table(out / "l2_metrics.parquet")
    for col in ECHO_COLS:
        assert col in l2.column_names, f"エコー列 {col} が {fire=} で欠けている"
    for col in PROPAGATION_COLS:
        assert col in l2.column_names, f"伝播列 {col} が {fire=} で欠けている"

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze_norms.py"), str(out),
         "--norm-stage", "2", "--norm-threshold", "2",
         "--out", str(tmp_path / f"norms_{int(fire)}.json")],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, \
        f"analyze_norms が {fire=} で失敗: {proc.stderr[-2000:]}"

    if fire:
        res = _analyze(out, tmp_path / f"fa_{int(fire)}")
        assert res["intervals"]["overall"]["n"] > 0
