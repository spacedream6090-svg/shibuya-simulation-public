"""W3-1: 凍結解析 3 本の最小修正(**2026-08-07 ユーザー承認**)の検収。

なぜこのテストが要るか
----------------------
`src/society/observer/metrics_spec.py` の `SPEC_FILES` 14 本は「事後に指標定義を
いじっていない」ことを `metrics_spec_hash` で証明するための凍結対象で、通常は
1 バイトも触らない。W3-1 は **8/15 の事前登録凍結の前に 1 回だけハッシュを動かして
確定し直す**ためにユーザーが明示承認した例外であり、動かしてよいのは次の 3 本だけ:

  - `scripts/analyze_beliefs.py`        … L1 の読み方(I/O)だけ
  - `scripts/analyze_norms.py`          … Δt(時間換算)の配管だけ
  - `scripts/analyze_specialization.py` … Δt(時間換算)の配管だけ

`scripts/diagnose_stationarity.py` と observer 側 9 本 + `truth_ledger.py` は
引き続きゼロタッチ(`tests/test_run_dt.py::test_frozen_spec_files_touched_only_
where_the_user_approved` が機械固定する)。

ここで固定するもの
------------------
  (1) **Δt=10 同値**: 修正前の実装を本ファイル内に **逐語温存**(`_legacy_*`)し、
      合成ランに対する出力が **JSON バイト同値**であること。判定式に 1 ミリも
      触っていないという主張の唯一の証拠。
  (2) **Δt=1 正当性**: norms の参入初日窓が 1440 step になる / specialization の
      日ビンが 1440 になる(= 直書き 144 が実際に外れている)。
  (3) beliefs の kind 絞り込み集合が **登録済みの belief_* 全種**と一致すること
      (将来 4 種目が生えたら気づける)。

実 LLM ランは使わない。合成 L1(parquet)だけで完結する。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "scripts", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze_beliefs as ab                             # noqa: E402
import analyze_norms as an                               # noqa: E402
import analyze_specialization as sp                      # noqa: E402
import run_dt                                            # noqa: E402

from society.observer import measure as m                # noqa: E402
from society.observer import metrics_spec as MS          # noqa: E402
from society.observer import schema as SCH               # noqa: E402


# --------------------------------------------------------------------------- #
# 合成ラン(test_run_dt.py::_write_run と同じ形。L1 の列と型は L1 の正準に合わせる)
# --------------------------------------------------------------------------- #
_L1_SCHEMA = pa.schema([("step", pa.int64()), ("sim_min", pa.int64()),
                        ("agent_id", pa.int64()), ("kind", pa.string()),
                        ("x", pa.float64()), ("y", pa.float64()),
                        ("payload", pa.string()), ("rng_stream", pa.string()),
                        ("llm_call_id", pa.string())])


def _write_run(tmp: Path, name: str, *, dt_min: int, events: list[dict],
               agents: list[dict] | None = None, n_steps: int = 0) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["run:", "  seed: 42", f"  n_agents: {len(agents or [])}",
             f"  n_steps: {n_steps or (max((e['step'] for e in events), default=0) + 1)}",
             f"  dt_min: {dt_min}", "  start_tod: 07:00", ""]
    (d / "config.yaml").write_text("\n".join(lines), encoding="utf-8")
    (d / "summary.json").write_text(
        json.dumps({"n_agents": len(agents or []), "n_events": len(events)}),
        encoding="utf-8")
    (d / "agents.json").write_text(json.dumps(agents or [], ensure_ascii=False),
                                   encoding="utf-8")
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e["step"]) * int(dt_min) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [0.0] * len(events),
        "y": [0.0] * len(events),
        "payload": [json.dumps(e.get("payload") or {}, ensure_ascii=False)
                    for e in events],
        "rng_stream": [None] * len(events),
        "llm_call_id": [None] * len(events),
    }
    pq.write_table(pa.Table.from_pydict(cols, schema=_L1_SCHEMA),
                   d / "l1_events.parquet")
    return d


def _ev(step, agent_id, kind, **payload):
    return {"step": step, "agent_id": agent_id, "kind": kind, "payload": payload}


# =========================================================================== #
# 1) analyze_beliefs.py … L1 の読み方だけを差し替えた(判定式ゼロタッチ)
# =========================================================================== #
_FACTS = [{"id": "f0", "value": 1.0, "kind": "event_host"},
          {"id": "f1", "value": 0.0, "kind": "venture_open"}]


def _belief_events() -> list[dict]:
    """belief 系 3 種 + 非 belief のノイズ + **未知の belief_ 種**を混ぜた合成 L1。

    未知種(`belief_foo`)を入れてあるのは、新実装が Arrow レベルで 3 種に絞ることが
    出力を変えないこと(下流 3 関数はいずれの経路でも未知種を読み飛ばす)を実際に
    確かめるため。
    """
    return [
        _ev(1, 1, "speak", text="ハチ公前は今日も混んでる"),
        _ev(2, 1, "belief_update", fact="f0", value=1.0, hop=0,
            verified="witness", cause="witness", **{"from": -1}),
        _ev(3, 1, "belief_transmit", fact="f0", to=[2, 3], hop=1, topic="t"),
        _ev(4, 2, "belief_update", fact="f0", value=0.8, hop=1,
            verified=None, cause="transmit", **{"from": 1}),
        _ev(5, 3, "belief_update", fact="f0", value=0.7, hop=1,
            verified=None, cause="transmit", **{"from": 1}),
        _ev(6, 2, "belief_verify", how="net", outcome="ok", match="topic",
            evidence="net"),
        _ev(7, 2, "belief_update", fact="f0", value=1.0, hop=1,
            verified="net", cause="verify", **{"from": -1}),
        _ev(8, 3, "belief_transmit", fact="f0", to=[4], hop=2, topic="t"),
        _ev(9, 4, "belief_update", fact="f1", value=0.3, hop=0,
            verified=None, cause="witness", **{"from": -1}),
        _ev(10, 5, "belief_foo", fact="f0", value=0.1),   # ★未知の belief_ 種
        _ev(11, 4, "hear", text="なにか"),
        _ev(12, 1, "belief_verify", how="go", outcome="pending", match="recent"),
        _ev(13, 4, "belief_update", fact="f1", value=0.0, hop=0,
            verified="site", cause="verify", **{"from": -1}),
        # 既に信念を持っている相手への再送 → 直後の更新が「訂正の伝播効率」に載る
        _ev(14, 1, "belief_transmit", fact="f0", to=[3], hop=1, topic="t"),
        _ev(15, 3, "belief_update", fact="f0", value=0.95, hop=1,
            verified=None, cause="transmit", **{"from": 1}),
        _ev(16, 3, "belief_transmit", fact="f0", to=[2], hop=2, topic="t"),
        _ev(17, 2, "belief_update", fact="f0", value=0.9, hop=2,
            verified=None, cause="transmit", **{"from": 3}),
    ]


def _beliefs_run(tmp_path) -> Path:
    d = _write_run(tmp_path, "bel", dt_min=10, events=_belief_events(),
                   agents=[{"id": i, "name": f"a{i}"} for i in range(6)])
    (d / ab.LEDGER_NAME).write_text(
        json.dumps({"n_facts": len(_FACTS), "facts": _FACTS}, ensure_ascii=False),
        encoding="utf-8")
    return d


def _legacy_analyze(run_dir: str, *, bin_steps: int = 24, top_k: int = 15) -> dict:
    """W3-1 **以前**の `analyze_beliefs.analyze()` の逐語温存。

    唯一の違いは L1 の読み方(`measure.load_events` で全件 RAM 展開 → Python 側で絞る)。
    下流(trace / verification / correction / tree_stats / 距離定義)は現行の実装を
    そのまま呼ぶので、**差が出たら I/O の置換が出力を変えた**ということになる。
    """
    ledger = ab.load_ledger(run_dir)
    facts = {f["id"]: f for f in ledger.get("facts", [])}
    truth = {fid: float(f["value"]) for fid, f in facts.items()}
    events = m.load_events(run_dir)                       # ★旧: 全件 RAM 展開
    bevents = ab.belief_events(events)

    tr = ab.trace(bevents, truth, bin_steps)
    ver = ab.verification(bevents, tr["final"])
    corr = ab.correction(bevents, truth)
    stats = ab.tree_stats(tr["tree"])

    kinds: dict[str, int] = defaultdict(int)
    for f in facts.values():
        kinds[str(f.get("kind"))] += 1
    n_bel = len(tr["final"])
    dist_final = (round(sum(s["dist"] for s in tr["final"].values()) / n_bel, 6)
                  if n_bel else 0.0)
    top = sorted(tr["by_agent"].items(),
                 key=lambda kv: (-kv[1]["n"], kv[0]))[:top_k]
    return {
        "run_dir": run_dir,
        "ledger": {"n_facts": len(facts), "by_kind": dict(sorted(kinds.items()))},
        "beliefs": {"n_agent_fact_pairs": n_bel,
                    "distance_mean_final": dist_final,
                    "n_updates": len(tr["updates"]),
                    "n_transmits": sum(1 for e in bevents
                                       if e["kind"] == "belief_transmit")},
        "series": tr["series"],
        "by_agent_top": [{"agent": a, **rec} for a, rec in top],
        "verification": ver,
        "correction": corr,
        "tree_stats": stats,
        "tree": tr["tree"],
        "holders": tr["holders"],
        "note": ab.NOTE,
    }


def _canon(doc) -> str:
    """`beliefs.json` の書き出しと同じ正規化(バイト同値の判定はこの文字列で行う)。"""
    return json.dumps(doc, sort_keys=True, ensure_ascii=False)


def test_beliefs_analyze_is_byte_identical_to_the_legacy_reader(tmp_path):
    """★Δt=10 同値: 逐次読みへの置換で `beliefs.json` が 1 バイトも変わらない。"""
    d = _beliefs_run(tmp_path)
    new = ab.analyze(str(d))
    old = _legacy_analyze(str(d))
    assert _canon(new) == _canon(old)
    # 中身が空でないこと(空同士の一致で通ってしまうのを防ぐ)
    assert new["beliefs"]["n_agent_fact_pairs"] > 0
    assert new["tree_stats"]["n_edges"] > 0
    assert new["verification"]["n_actions"] == 2
    assert new["correction"]["n_from_verified"] + \
        new["correction"]["n_from_unverified"] > 0


def test_beliefs_analyze_matches_legacy_for_all_bin_widths(tmp_path):
    """ビン幅を変えても同値(時系列 series は読み方の影響を最も受けやすい)。"""
    d = _beliefs_run(tmp_path)
    for bs in (1, 3, 24, 144):
        assert _canon(ab.analyze(str(d), bin_steps=bs)) \
            == _canon(_legacy_analyze(str(d), bin_steps=bs)), f"bin_steps={bs}"


def test_load_belief_events_equals_the_legacy_filter(tmp_path):
    """絞り込みの結果が「全件読んでから絞ったもの」と**同じ列・同じ順序**。"""
    d = _beliefs_run(tmp_path)
    legacy = ab.belief_events(m.load_events(str(d)))
    got = ab.load_belief_events(str(d))
    # 未知種 belief_foo は旧経路にだけ入る(それが出力を変えないことは上のテストが固定)
    assert [e["kind"] for e in legacy if e["kind"] in ab.BELIEF_KINDS] == \
        [e["kind"] for e in got]
    assert [e for e in legacy if e["kind"] in ab.BELIEF_KINDS] == got


def test_belief_kinds_covers_every_registered_belief_kind():
    """★`BELIEF_KINDS` が登録済みの belief_* を**全部**覆っている(4 種目の見落とし検知)。"""
    registered = {k for k in SCH.EVENT_KINDS if k.startswith("belief_")}
    assert registered == set(ab.BELIEF_KINDS), \
        f"登録種と絞り込み集合がずれている: {registered} vs {set(ab.BELIEF_KINDS)}"


def test_beliefs_does_not_load_all_events_anymore():
    """L1 全件展開の経路がコードから消えていること(**実際に評価される識別子**で判定)。

    散文(docstring / コメント)は対象外 — 旧経路の説明は残しておきたいので、
    文字列一致ではなく AST で見る(`tests/test_rejection_notify.py` と同じ流儀)。
    """
    import ast
    tree = ast.parse((_ROOT / "scripts" / "analyze_beliefs.py")
                     .read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "load_events" not in attrs and "load_events" not in names, \
        "全件 RAM 展開(measure.load_events)が残っている"
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    assert "l1_stream" in imported, "共有ストリーミング読みを借りていない"
    assert "iter_events" in attrs and "l1_paths" in attrs


def test_beliefs_refuses_a_run_without_l1(tmp_path):
    """L1 が無いランは**黙って 0 を返さない**(欠測を偽の値で埋めない)。"""
    d = tmp_path / "empty"
    d.mkdir()
    try:
        ab.load_belief_events(str(d))
    except SystemExit as exc:
        assert "l1_events.parquet" in str(exc)
    else:
        raise AssertionError("L1 が無いのに黙って通した")


def test_beliefs_has_no_dt_conversion_change():
    """beliefs は **時間換算の配管には触っていない**(承認範囲は I/O だけ)。"""
    src = (_ROOT / "scripts" / "analyze_beliefs.py").read_text(encoding="utf-8")
    assert "run_dt" not in src


# =========================================================================== #
# 2) analyze_norms.py … 参入初日窓の 1 日を run 由来へ
# =========================================================================== #
def _norm_events(spd: int) -> list[dict]:
    """同じ**実時間**の並びを Δt に合わせて作る(1 日 = spd step)。

    - agent 1/2 は step 0 参入。初日の**後半**(0.9 日め)にも発話する
      → 窓が 144 のままだと Δt=1 ランでその発話が初日から落ちる。
    - agent 3 は 1.5 日めに初参入(post コホート側の材料)。
    """
    late = int(spd * 0.9)
    out = [
        _ev(0, 1, "speak", text="ハチ公前に行こう"),
        _ev(0, 2, "speak", text="ハチ公前に行こう"),
        _ev(1, 1, "stay"),
        _ev(2, 2, "stay"),
        _ev(late, 1, "speak", text="ハチ公前は混んでいる"),
        _ev(late, 2, "stay"),
        _ev(int(spd * 1.5), 3, "speak", text="ハチ公前ってどこ"),
        _ev(int(spd * 1.6), 3, "stay"),
        _ev(int(spd * 2.0) - 1, 1, "stay"),
    ]
    return sorted(out, key=lambda e: e["step"])


def _norms_run(tmp_path, name: str, dt_min: int) -> Path:
    spd = 1440 // dt_min
    return _write_run(tmp_path, name, dt_min=dt_min, events=_norm_events(spd),
                      agents=[{"id": i, "name": f"a{i}", "occupation": "会社員"}
                              for i in range(1, 4)])


def _legacy_analyze_run(run_dir: str, min_stage: int, min_agents: int,
                        require_full_day: bool = False) -> dict:
    """W3-1 **以前**の `analyze_norms.analyze_run()` の逐語温存。

    唯一の違いは `first_day_features` に `spd` を渡さないこと(= 直書き 144)。
    """
    mcfg, msrc = an.load_markers(run_dir)
    res = an.S.norm_stages(run_dir, mcfg["markers"], mcfg["min_word_len"],
                           mcfg["max_gap"], usage=True)
    cands = an.N.norm_candidates(res, min_stage, min_agents, res.get("usage"))
    t_norm = cands[0]["established_step"] if cands else None
    words = [c["word"] for c in cands]
    first = an.S.first_presence(run_dir)
    feats, last_step = an.first_day_features(run_dir, first, words, t_norm)
    n_truncated = sum(1 for r in feats if not r["full_day"])
    if require_full_day:
        feats = [r for r in feats if r["full_day"]]
    cohorts = {"pre": [r for r in feats if r["cohort"] == "pre"],
               "post": [r for r in feats if r["cohort"] == "post"]}
    return {
        "dir": run_dir, "name": os.path.basename(os.path.normpath(run_dir)),
        "marker_source": msrc,
        "initial_frame": an._read_initial_frame(run_dir),
        "stages": res["summary"],
        "words": res["words"],
        "candidates": cands,
        "t_norm": t_norm,
        "n_agents": len(first),
        "n_first_steps_distinct": len(set(first.values())),
        "last_step": last_step,
        "n_truncated_first_day": n_truncated,
        "require_full_day": bool(require_full_day),
        "cohort_sizes": {k: len(v) for k, v in cohorts.items()},
        "truncated_by_cohort": {
            k: sum(1 for r in v if not r["full_day"]) for k, v in cohorts.items()},
        "features": feats,
    }


def test_norms_dt10_is_byte_identical_to_the_legacy_window(tmp_path):
    """★Δt=10 同値: run 由来に変えても 144 に解決され、出力が 1 バイトも変わらない。"""
    d = _norms_run(tmp_path, "norm10", dt_min=10)
    assert run_dt.steps_per_day(str(d)) == an.STEPS_PER_DAY == 144
    for full in (False, True):
        new = an.analyze_run(str(d), 2, 2, require_full_day=full)
        old = _legacy_analyze_run(str(d), 2, 2, require_full_day=full)
        assert _canon(new) == _canon(old), f"require_full_day={full}"
    assert an.analyze_run(str(d), 2, 2)["features"], "特徴が空(前提が崩れた)"


def test_norms_first_day_window_follows_dt(tmp_path):
    """★Δt=1 正当性: 初日窓が **1440 step** になる(直書き 144 が外れている)。"""
    d = _norms_run(tmp_path, "norm1", dt_min=1)
    assert run_dt.steps_per_day(str(d)) == 1440
    new = an.analyze_run(str(d), 2, 2)
    old = _legacy_analyze_run(str(d), 2, 2)      # 直書き 144 のまま
    by_new = {r["agent"]: r for r in new["features"]}
    by_old = {r["agent"]: r for r in old["features"]}
    # 0.9 日め(step 1296)の行動は 1440 窓には入り、144 窓には入らない
    assert by_new[1]["n_events"] > by_old[1]["n_events"], \
        "窓が広がっていない(144 が直書きのまま)"
    assert by_new[1]["observed_steps"] == 1440 and by_new[1]["full_day"] is True
    assert by_old[1]["observed_steps"] == 144
    # 「1.5 日めに参入」は 1440 窓では first_day=1、144 窓では 15 と誤読される
    assert by_new[3]["first_day"] == 1 and by_old[3]["first_day"] == 15


def test_norms_first_day_features_default_is_the_canonical_144(tmp_path):
    """`spd` を渡さなければ従来と完全同値(既定引数が正準 144)。"""
    d = _norms_run(tmp_path, "norm_def", dt_min=1)
    first = an.S.first_presence(str(d))
    a = an.first_day_features(str(d), first, [], None)
    b = an.first_day_features(str(d), first, [], None, 144)
    c = an.first_day_features(str(d), first, [], None, 1440)
    assert a == b and a != c


def test_norms_cli_reports_run_derived_steps_per_day(tmp_path):
    """CLI の `params.steps_per_day` が **ラン由来**になる(Δt=10 では従来と同値)。"""
    for dt, want in ((10, 144), (1, 1440)):
        d = _norms_run(tmp_path, f"cli{dt}", dt_min=dt)
        out = tmp_path / f"norms{dt}.json"
        r = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "analyze_norms.py"), str(d),
             "--norm-stage", "2", "--norm-threshold", "2", "--mc", "50",
             "--out", str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(_ROOT))
        assert r.returncode == 0, r.stderr
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["params"]["steps_per_day"] == want, f"dt_min={dt}"


# =========================================================================== #
# 3) analyze_specialization.py … 日ビン幅の既定を run 由来へ
# =========================================================================== #
def _spec_events(spd: int) -> list[dict]:
    """3 日ぶんの発話(役割 2 群 × 定着語)。日ビン幅が効くよう日をまたいで撒く。"""
    out: list[dict] = []
    for day in range(3):
        base = day * spd
        for k in range(4):
            out.append(_ev(base + k * 3 + 1, 1, "speak",
                           text="しんりょうしんりょうカルテカルテ"))
            out.append(_ev(base + k * 3 + 2, 2, "speak",
                           text="しんりょうしんりょうカルテカルテ"))
            out.append(_ev(base + k * 3 + 3, 3, "speak",
                           text="こうぎこうぎレポートレポート"))
            out.append(_ev(base + k * 3 + 4, 4, "speak",
                           text="こうぎこうぎレポートレポート"))
    return sorted(out, key=lambda e: e["step"])


def _spec_run(tmp_path, name: str, dt_min: int) -> Path:
    spd = 1440 // dt_min
    agents = [{"id": 1, "occupation": "医師"}, {"id": 2, "occupation": "医師"},
              {"id": 3, "occupation": "学生"}, {"id": 4, "occupation": "学生"}]
    return _write_run(tmp_path, name, dt_min=dt_min, events=_spec_events(spd),
                      agents=agents)


def _spec_cli(run_dir: Path, out: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "analyze_specialization.py"),
         str(run_dir), "--min-delta", "0.02", "--alpha", "0.05",
         "--min-count", "2", "--min-users", "2", "--ttr-budget", "8",
         "--min-cluster", "2", "--min-role-tokens", "4", "--max-df", "1.0",
         "--out", str(out), *[str(x) for x in extra]],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(_ROOT))


def test_specialization_dt10_default_is_byte_identical_to_explicit_144(tmp_path):
    """★Δt=10 同値: `--bin-steps` 省略 = 従来の既定 144 と **JSON バイト一致**。"""
    d = _spec_run(tmp_path, "spec10", dt_min=10)
    a, b = tmp_path / "outA", tmp_path / "outB"
    r1 = _spec_cli(d, a)
    r2 = _spec_cli(d, b, "--bin-steps", "144")
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    ja = (a / "specialization.json").read_bytes()
    jb = (b / "specialization.json").read_bytes()
    assert ja == jb, "既定(run 由来)と旧既定 144 で出力が違う"
    doc = json.loads(ja.decode("utf-8"))
    assert doc["params"]["bin_steps"] == 144
    # ③ が実際に算出されている(None 同士の一致で通ってしまうのを防ぐ)
    assert doc["treatment"]["means"]["vocab_persistence"] is not None
    assert (a / "specialization_report.md").read_bytes() \
        == (b / "specialization_report.md").read_bytes()


def test_specialization_bin_steps_follows_dt(tmp_path):
    """★Δt=1 正当性: ビン幅が **1440** に解決され、144 とは違う結果になる。"""
    d = _spec_run(tmp_path, "spec1", dt_min=1)
    a, b = tmp_path / "out1A", tmp_path / "out1B"
    r1 = _spec_cli(d, a)
    r2 = _spec_cli(d, b, "--bin-steps", "144")
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    da = json.loads((a / "specialization.json").read_text(encoding="utf-8"))
    db = json.loads((b / "specialization.json").read_text(encoding="utf-8"))
    assert da["params"]["bin_steps"] == 1440
    assert db["params"]["bin_steps"] == 144
    assert da["treatment"]["means"]["vocab_persistence"] \
        != db["treatment"]["means"]["vocab_persistence"], \
        "ビン幅が結果に効いていない(配管が死んでいる)"


def test_specialization_module_constant_is_still_the_canonical_144():
    """モジュール定数は残す(下位関数の既定・help の表示に使う)。値は正準 144。"""
    assert sp.STEPS_PER_DAY == run_dt.CANON_STEPS_PER_DAY == 144


def test_specialization_scores_are_untouched_by_this_batch(tmp_path):
    """下位スコア 3 本の**呼び出し規約**は不変(bin_steps を明示すれば従来どおり)。"""
    recs = [{"step": d * 144 + 1, "agent": d % 2, "tokens": ["ああ", "あい"]}
            for d in range(5)]
    out = sp.vocab_persistence(recs, 4 * 144 + 1, min_count=2, min_users=2,
                               bin_steps=144)
    assert out["value"] == 1.0


# =========================================================================== #
# 4) 凍結の状態(ハッシュは live 計算・リテラル固定はしない)
# =========================================================================== #
def test_the_three_approved_files_are_still_in_the_frozen_list():
    """承認された 3 本は**凍結リストから外していない**(外す改竄も hash に出る規約)。"""
    approved = {"scripts/analyze_beliefs.py", "scripts/analyze_norms.py",
                "scripts/analyze_specialization.py"}
    assert approved <= set(MS.SPEC_FILES)
    assert len(MS.SPEC_FILES) == 14
    assert MS.compute()["missing"] == []


def test_stationarity_and_observer_side_are_untouched():
    """★承認外(diagnose_stationarity + observer 側 9 本 + truth_ledger)はゼロタッチ。

    「触っていない」ことをテストで直接証明はできない(HEAD との比較は commit 後に
    腐る)ので、ここでは **W3-1 が持ち込んだ 2 つの識別子**が 1 つも現れないことを
    固定する(= 触っていたら必ずどちらかが出る形の変更しかしていない)。
    """
    untouched = [f for f in MS.SPEC_FILES
                 if f not in ("scripts/analyze_beliefs.py",
                              "scripts/analyze_norms.py",
                              "scripts/analyze_specialization.py")]
    assert len(untouched) == 11
    for rel in untouched:
        src = (_ROOT / rel).read_text(encoding="utf-8")
        assert "run_dt" not in src, f"承認外の {rel} に W3-1 の Δt 対応が入っている"
        assert "l1_stream" not in src, f"承認外の {rel} に W3-1 の I/O 変更が入っている"
