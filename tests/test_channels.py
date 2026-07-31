"""第80バッチ: 観測チャンネル o_c(t) + σ_c 実測ハーネス + 較正テーブル外部化のテスト。

正典: docs/plans/source/cognition-design-record.md §2.3-2.4 /
      docs/plans/source/physics-instructions.md Part P0(3) /
      docs/plans/cognition-physics-plan.md §4 第80行。

守るもの(検収基準の順)
  (1) 既定 OFF = サイドカー不在・manifest にキーなし。ON でも **L1/L2/L3 バイト一致**・
      乱数 draw 一致・LLM 呼数一致(観測はシムを変えない)
  (2) σ_c: mock パイロットから凍結ファイルを作れる・**同入力2回でバイト同一**・
      σ=0 は usable=false で **除外**(床を敷かない)
  (3) パリティ: channels.parquet が L1 / L3 から**独立に検算できる**
      (同席数・遭遇数・受信発話数を観測層のコードを使わずに再計算して一致)
  (4) resume: 分割実行しても行が二重にならない(split == straight)
  (5) 宣言: registry(repro_tier)と timeconv(Δt 分類)に新キーが載っている
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import registry as R
from society import timeconv as T
from society.cognition import calib as CALIB
from society.cognition import channels as CH
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SIGMA = REPO_ROOT / CALIB.SIGMA_DEFAULT_REL
SHIPPED_CALIB = REPO_ROOT / CALIB.CALIB_DEFAULT_REL


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int = 24, n_agents: int = 24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 24, n_agents: int = 24, **ov):
    out = tmp_path / name
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=out)
    sim.run()
    return sim, out


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _rows(out_dir: Path, stem: str):
    path = out_dir / f"{stem}.parquet"
    return pq.read_table(path).to_pylist() if path.exists() else None


class _CountingHub:
    """stream 派生と draw 消費を数えるプロキシ(test_weather_generated と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


# --------------------------------------------------------------------------- #
# (A) チャンネル定義の健全性
# --------------------------------------------------------------------------- #
def test_channel_registry_is_well_formed():
    ids = [c.id for c in CH.CHANNELS]
    cols = [c.column for c in CH.CHANNELS]
    assert len(ids) == len(set(ids)), "チャンネル id が重複している"
    assert len(cols) == len(set(cols)), "列名が重複している"
    assert not (set(cols) & set(CH.KEY_COLUMNS)), "値列が固定キー列と衝突している"
    for c in CH.CHANNELS:
        assert c.source in CH.SOURCES, f"{c.id}: 未知の源 {c.source}"
        assert c.description.strip(), f"{c.id}: 説明が空"
        assert c.column == c.id.replace(".", "_")
    # 設計 §2.4 の 3 源がすべて登録されている
    assert {c.source for c in CH.CHANNELS} == set(CH.SOURCES)


def test_prediction_channel_is_declared_but_not_implemented():
    """予測不成立は**枠だけ**(第81 で ô が入るまで値は未使用)。"""
    pred = [c for c in CH.CHANNELS if c.source == CH.PREDICTION]
    assert pred, "予測不成立チャンネルの枠が無い"
    assert all(not c.implemented for c in pred), \
        "本バッチでは予測不成立チャンネルは未実装のままでなければならない"


def test_spec_hash_is_stable_and_binds_the_definition():
    a, b = CH.spec_sha256(), CH.spec_sha256()
    assert a == b and len(a) == 64
    assert len(CH.describe()) == len(CH.CHANNELS)


def test_columns_order_matches_observe_row_width(tmp_path):
    """observe() の値の並びが CHANNELS の並びと厳密に一致する(= 列がずれない)。"""
    sim = Simulation(_cfg("ord", n_steps=1), out_dir=tmp_path / "ord")
    rows = CH.observe(sim, 0, 0, 0)
    assert rows and len(rows[0]) == len(CH.KEY_COLUMNS) + len(CH.CHANNELS)
    ids = [int(r[2]) for r in rows]
    assert ids == sorted(ids), "行が agent_id 昇順で出ていない(決定論の前提)"


def test_missing_sources_are_null_not_zero(tmp_path):
    """元機構が無効なチャンネルは **null**(0 で埋めない=欠測を欠測と判る値にする)。"""
    sim = Simulation(_cfg("null", n_steps=1), out_dir=tmp_path / "null")
    row = CH.observe(sim, 0, 0, 0)[0]
    idx = {c.id: len(CH.KEY_COLUMNS) + i for i, c in enumerate(CH.CHANNELS)}
    assert row[idx["ext.weather_temp"]] is None, "天候 OFF なのに気温が捏造されている"
    assert row[idx["body.boredom"]] is None, "退屈ゲージ OFF なのに値が捏造されている"
    assert row[idx["pred.unmet"]] is None, "未実装チャンネルに値が入っている"


# --------------------------------------------------------------------------- #
# (B) 既定 OFF / ON の非侵襲性(検収基準 1)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.cognition.channels.enabled) is False
    assert int(cfg.cognition.channels.every_steps) == 1


def test_off_writes_nothing(tmp_path):
    sim, out = _run(tmp_path, "ch_off", n_steps=12, n_agents=10)
    assert sim.channels_sc is None
    assert sim.cognition_calib is None and sim.cognition_sigma is None
    assert not (out / "channels.parquet").exists()
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert "cognition" not in man, "OFF なのに manifest に来歴キーが生えた"


def test_on_leaves_l1_l2_l3_and_rng_and_llm_untouched(tmp_path):
    """★検収基準 1: ON でも世界は 1 バイトも変わらない(サイドカーが増えるだけ)。"""
    off = Simulation(_cfg("inv_off", 48, 30), out_dir=tmp_path / "inv_off")
    off.hub = _CountingHub(off.hub)
    off.run()
    on = Simulation(_cfg("inv_on", 48, 30, **{"cognition.channels.enabled": "true"}),
                    out_dir=tmp_path / "inv_on")
    on.hub = _CountingHub(on.hub)
    on.run()

    assert _l1(off) == _l1(on), "channels ON で L1 が変わった"
    assert [dict(m) for m in off.logger.metrics] == [dict(m) for m in on.logger.metrics], \
        "channels ON で L2 が変わった"
    assert [dict(s) for s in off.logger.snapshots] == \
           [dict(s) for s in on.logger.snapshots], "channels ON で L3 が変わった"
    assert off.hub.counts == on.hub.counts, \
        f"乱数 stream の消費が変わった: {off.hub.counts} vs {on.hub.counts}"
    assert off.llm.calls == on.llm.calls > 0, \
        f"LLM 呼数が変わった: {off.llm.calls} vs {on.llm.calls}"
    # 増えるのはサイドカー 1 本だけ
    assert (tmp_path / "inv_on" / "channels.parquet").exists()
    assert not (tmp_path / "inv_off" / "channels.parquet").exists()


def test_on_is_deterministic_across_two_runs(tmp_path):
    _run(tmp_path, "det_a", 24, 20, **{"cognition.channels.enabled": "true"})
    _run(tmp_path, "det_b", 24, 20, **{"cognition.channels.enabled": "true"})
    a = pq.read_table(tmp_path / "det_a" / "channels.parquet").to_pylist()
    b = pq.read_table(tmp_path / "det_b" / "channels.parquet").to_pylist()
    assert a == b and a, "同 seed 2 ランで channels が不一致"


# --------------------------------------------------------------------------- #
# (C) 記録の形
# --------------------------------------------------------------------------- #
def test_sidecar_schema_and_row_count(tmp_path):
    _sim, out = _run(tmp_path, "sc", 12, 10, **{"cognition.channels.enabled": "true"})
    table = pq.read_table(out / "channels.parquet")
    assert table.column_names == list(CH.KEY_COLUMNS) + list(CH.COLUMNS)
    assert table.num_rows == 12 * 10, "全 step × 全 agent が 1 行ずつ出ていない"
    assert sorted(set(table.column("step").to_pylist())) == list(range(12))
    # 未実装チャンネルの列は存在するが全 null(= 第81 の枠)
    assert all(v is None for v in table.column("pred_unmet").to_pylist())


def test_every_steps_thins_the_sampling(tmp_path):
    _sim, out = _run(tmp_path, "thin", 12, 8,
                     **{"cognition.channels.enabled": "true",
                        "cognition.channels.every_steps": 4})
    steps = sorted(set(pq.read_table(out / "channels.parquet")
                       .column("step").to_pylist()))
    assert steps == [0, 4, 8], f"採取 step が間引かれていない: {steps}"


# --------------------------------------------------------------------------- #
# (D) パリティ: L1 / L3 から独立に検算できる(検収基準 3)
#
#  ★ここでは観測層(cognition/channels.py)のコードを一切呼ばず、成果物(L1/L3)だけから
#    定義どおりに数え直す。観測層のバグが「同じバグで検算」されないようにするため。
# --------------------------------------------------------------------------- #
def _l3_states(out_dir: Path) -> dict[int, list[dict]]:
    rows = _rows(out_dir, "l3_snapshots") or []
    return {int(r["step"]): json.loads(r["state"])["agents"] for r in rows}


def _channels_by_step(out_dir: Path) -> dict[int, dict[int, dict]]:
    table = pq.read_table(out_dir / "channels.parquet").to_pylist()
    out: dict[int, dict[int, dict]] = {}
    for row in table:
        out.setdefault(int(row["step"]), {})[int(row["agent_id"])] = row
    return out


def test_crowd_local_recomputed_from_l3_matches(tmp_path):
    """★検収基準 3: 局所混雑度 = 同席数 を L3 スナップショットから独立に再計算して一致。"""
    _sim, out = _run(tmp_path, "par_crowd", 36, 40,
                     **{"cognition.channels.enabled": "true",
                        "observer.snapshot_every": 6})
    chans = _channels_by_step(out)
    states = _l3_states(out)
    assert states, "L3 スナップショットが無い"
    checked = 0
    for step, agents in states.items():
        counts: dict = {}
        for a in agents:
            if a["loc"] == "outside":
                continue
            key = (("bld", a["building"], int(a["floor"])) if a["building"]
                   else ("node", a["node"]))
            counts[key] = counts.get(key, 0) + 1
        for a in agents:
            if a["loc"] == "outside":
                want = 0.0
            else:
                key = (("bld", a["building"], int(a["floor"])) if a["building"]
                       else ("node", a["node"]))
                want = float(counts[key] - 1)
            got = chans[step][int(a["id"])]["ext_crowd_local"]
            assert got == pytest.approx(want), \
                f"step {step} agent {a['id']}: 同席数が L3 と不一致 {got} != {want}"
            checked += 1
    assert checked > 0
    # 「全部 0 で偶然一致」ではないことを固定する
    assert any(r["ext_crowd_local"] > 0
               for by_agent in chans.values() for r in by_agent.values()), \
        "同席がゼロのランで検算しても意味がない(人数/step を増やすこと)"


def test_encounter_recomputed_from_l3_matches(tmp_path):
    """遭遇数 = 知覚半径内の起きている他者。L3 の座標から独立に再計算して一致。"""
    cfg_radius = float(load_config().world.perception_radius_m)
    _sim, out = _run(tmp_path, "par_enc", 36, 40,
                     **{"cognition.channels.enabled": "true",
                        "observer.snapshot_every": 6})
    chans = _channels_by_step(out)
    for step, agents in _l3_states(out).items():
        def ctx(a):
            if a["loc"] == "outside":
                return ("outside", a["id"])
            if a["building"]:
                return ("bld", a["building"], int(a["floor"]))
            return ("street",)

        for a in agents:
            if a["loc"] == "outside":
                want = 0
            else:
                want = sum(
                    1 for b in agents
                    if b["id"] != a["id"] and not b["sleeping"]
                    and b["loc"] != "outside" and ctx(b) == ctx(a)
                    and math.hypot(b["x"] - a["x"], b["y"] - a["y"]) <= cfg_radius)
            got = chans[step][int(a["id"])]["ext_encounter"]
            assert got == pytest.approx(float(want)), \
                f"step {step} agent {a['id']}: 遭遇数が L3 と不一致 {got} != {want}"


def test_heard_recomputed_from_l1_matches(tmp_path):
    """受信発話数 = L1 の hear(自分宛)+ dm(payload.to が自分)を独立に数えて一致。"""
    _sim, out = _run(tmp_path, "par_heard", 48, 40,
                     **{"cognition.channels.enabled": "true"})
    want: dict[tuple[int, int], int] = {}
    for row in _rows(out, "l1_events"):
        payload = json.loads(row["payload"])
        if row["kind"] == "hear" and int(row["agent_id"]) >= 0:
            key = (int(row["step"]), int(row["agent_id"]))
            want[key] = want.get(key, 0) + 1
        elif row["kind"] == "dm" and payload.get("to") is not None:
            key = (int(row["step"]), int(payload["to"]))
            want[key] = want.get(key, 0) + 1
    assert want, "hear/dm が 1 件も出ていない(検算にならない)"
    chans = _channels_by_step(out)
    for (step, aid), n in want.items():
        got = chans[step][aid]["ext_heard"]
        assert got == pytest.approx(float(n)), \
            f"step {step} agent {aid}: 受信発話数が L1 と不一致 {got} != {n}"
    # 逆向き: 数えた以外の (step, agent) は 0 でなければならない
    for step, by_agent in chans.items():
        for aid, row in by_agent.items():
            if (step, aid) not in want:
                assert row["ext_heard"] == 0.0, \
                    f"step {step} agent {aid}: L1 に無い受信が記録されている"


# --------------------------------------------------------------------------- #
# (E) resume で二重記録しない(検収基準 4)
# --------------------------------------------------------------------------- #
_ON = {"cognition.channels.enabled": "true"}


def test_resume_crash_recovery_matches_straight(tmp_path):
    """途中で落ちたランを再開しても straight と行が完全一致(重複も欠落もない)。"""
    straight = tmp_path / "straight"
    Simulation(_cfg("straight", 24, 24, **_ON), out_dir=straight).run()

    d = tmp_path / "resumed"
    every = {"observer.checkpoint_every": 12}
    sim1 = Simulation(_cfg("resumed", 12, 24, **_ON, **every), out_dir=d)
    for step in range(12):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 12, d / "checkpoint" / "ckpt-000012.pkl.gz")
    sim1.logger.flush_segment()
    sim1.channels_sc.flush_segment()
    sim2 = Simulation(_cfg("resumed", 24, 24, **_ON, **every), out_dir=d)
    sim2.run(resume_from=d)

    assert _rows(straight, "l1_events") == _rows(d, "l1_events"), "L1 不一致"
    assert _rows(straight, "channels") == _rows(d, "channels"), "channels 不一致"


def test_resume_split_execution_does_not_duplicate(tmp_path):
    """分割実行(chunk1 が clean finalize → chunk2 が resume)でも行が二重にならない。"""
    straight = tmp_path / "s2"
    Simulation(_cfg("s2", 24, 24, **_ON), out_dir=straight).run()

    d = tmp_path / "split"
    every = {"observer.checkpoint_every": 12}
    Simulation(_cfg("split", 12, 24, **_ON, **every), out_dir=d).run()
    first = _rows(d, "channels")
    assert len(first) == 12 * 24
    Simulation(_cfg("split", 24, 24, **_ON, **every), out_dir=d).run(resume_from=d)

    got = _rows(d, "channels")
    assert len(got) == 24 * 24, f"分割実行で行数がおかしい: {len(got)}"
    seen = {(r["step"], r["agent_id"]) for r in got}
    assert len(seen) == len(got), "同じ (step, agent) が二重に記録されている"
    assert got == _rows(straight, "channels"), "split が straight と不一致"


# --------------------------------------------------------------------------- #
# (F) 較正テーブルの外部化(P0-3)
# --------------------------------------------------------------------------- #
def test_shipped_calib_table_loads_and_validates():
    loaded = CALIB.load_calib(None)
    assert Path(loaded["path"]) == SHIPPED_CALIB
    assert len(loaded["sha256"]) == 64
    table = loaded["table"]
    assert tuple(table["contexts"]) == CALIB.CONTEXTS
    assert tuple(table["stimuli"]) == CALIB.STIMULI
    for ctx in CALIB.CONTEXTS:
        assert table["base_period"][ctx]["mean_min"] > 0
        for sti in CALIB.STIMULI:
            cell = table["onset_delay"][ctx][sti]
            assert cell["dist"] in CALIB.DISTS
            assert cell["median_s"] > 0 and cell["sigma"] > 0


def test_shipped_calib_is_declared_provisional():
    """★実測に合わせ込む前の仮置きであることがファイル自身に書いてあること。"""
    assert CALIB.load_calib(None)["table"]["status"] == "provisional"


def test_calib_hash_changes_when_the_table_changes(tmp_path):
    body = SHIPPED_CALIB.read_text(encoding="utf-8")
    same = tmp_path / "same.yaml"
    same.write_text(body, encoding="utf-8")
    assert CALIB.file_sha256(same) == CALIB.file_sha256(SHIPPED_CALIB), \
        "同一内容なのに hash が違う(改行正規化が効いていない)"
    changed = tmp_path / "changed.yaml"
    changed.write_text(body.replace("mean_min: 8.0", "mean_min: 9.0"), encoding="utf-8")
    assert CALIB.file_sha256(changed) != CALIB.file_sha256(SHIPPED_CALIB)


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("base_period"),
    lambda d: d["base_period"].pop("walking"),
    lambda d: d["onset_delay"]["walking"].pop("social"),
    lambda d: d["onset_delay"]["walking"]["social"].update({"dist": "uniform"}),
    lambda d: d["onset_delay"]["walking"]["social"].update({"median_s": -1.0}),
    lambda d: d["base_period"]["resting"].update({"cv": 0.0}),
    lambda d: d.update({"schema": 99}),
    lambda d: d.update({"contexts": ["walking"]}),
])
def test_broken_calib_table_raises(mutate):
    """壊れたテーブルは**黙って既定へ落ちない**(凍結値で時定数が決まる層なので)。"""
    from omegaconf import OmegaConf
    doc = OmegaConf.to_container(OmegaConf.load(SHIPPED_CALIB), resolve=True)
    mutate(doc)
    with pytest.raises(ValueError):
        CALIB.validate_calib(doc, "test")


def test_manifest_records_calib_and_sigma_provenance(tmp_path):
    """★P0-3: テーブルのハッシュが manifest に載る(天候来歴と同じ流儀)。"""
    _sim, out = _run(tmp_path, "prov", 6, 8, **_ON)
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    cog = man["cognition"]
    assert cog["channel_spec_sha256"] == CH.spec_sha256()
    assert cog["n_channels"] == len(CH.CHANNELS)
    assert cog["calib"]["sha256"] == CALIB.file_sha256(SHIPPED_CALIB)
    assert cog["calib"]["status"] == "provisional"
    assert cog["sigma_c"]["status"] in ("loaded", "absent")
    # 来歴のパスはリポジトリ相対(実行環境のユーザー名を成果物に混ぜない)
    for key in ("calib", "sigma_c"):
        assert not Path(cog[key]["file"]).is_absolute(), \
            f"{key} のパスが絶対パスで記録されている: {cog[key]['file']}"


# --------------------------------------------------------------------------- #
# (G) σ_c 実測ハーネス(検収基準 2)
# --------------------------------------------------------------------------- #
def _measure(tmp_path, out_name: str, *, agents=8, steps=12, extra=()):
    out = tmp_path / out_name
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "measure_sigma.py"),
           "--agents", str(agents), "--steps", str(steps),
           "--run-dir", str(tmp_path / f"{out_name}_run"), "--out", str(out),
           *extra]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900)
    assert proc.returncode == 0, f"measure_sigma が失敗: {proc.stderr[-2000:]}"
    return out


def test_measure_sigma_is_deterministic(tmp_path):
    """★検収基準 2: 同入力 2 回でバイト同一(壁時計を書かない)。"""
    a = _measure(tmp_path, "sig_a.json")
    b = _measure(tmp_path, "sig_b.json")
    assert a.read_bytes() == b.read_bytes(), "measure_sigma が非決定"


def test_measure_sigma_output_is_loadable_and_excludes_constant_channels(tmp_path):
    path = _measure(tmp_path, "sig_c.json")
    loaded = CALIB.load_sigma(str(path))            # payload_sha256 の照合込み
    assert loaded["status"] == "loaded"
    doc = loaded["doc"]
    assert doc["meta"]["channel_spec_sha256"] == CH.spec_sha256()
    assert set(doc["channels"]) == {c.id for c in CH.CHANNELS}
    # σ=0(定数)/ 未実装 / 標本なし は usable=false → sigma_of から**除外**される
    usable = CALIB.sigma_of(loaded)
    for cid, row in doc["channels"].items():
        if row["usable"]:
            assert usable[cid] > 0.0
        else:
            assert cid not in usable, f"{cid}: usable=false なのに σ が配られている"
            assert row["reason"], f"{cid}: 除外理由が書かれていない"
    assert "pred.unmet" not in usable, "未実装チャンネルが σ を持っている"
    assert usable, "usable なチャンネルが 1 本も無い(パイロットが小さすぎる)"
    # 床は敷かない(方針は残すが適用しない)= 除外方式であることの固定
    assert doc["meta"]["policy"]["zero_sigma"].startswith("excluded")


def test_tampered_sigma_file_is_rejected(tmp_path):
    path = _measure(tmp_path, "sig_d.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    first = sorted(doc["channels"])[0]
    doc["channels"][first]["sigma"] = 999.0          # payload だけ書き換える
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        CALIB.load_sigma(str(bad))


def test_absent_sigma_file_does_not_break_the_run(tmp_path):
    """σ ファイルが無くてもランは落ちない(σ を測るパイロット自体が無い状態で回る)。"""
    loaded = CALIB.load_sigma(str(tmp_path / "nope.json"))
    assert loaded["status"] == "absent" and CALIB.sigma_of(loaded) == {}
    with pytest.raises(FileNotFoundError):
        CALIB.load_sigma(str(tmp_path / "nope.json"), required=True)


@pytest.mark.skipif(not FROZEN_SIGMA.exists(), reason="σ_c 凍結ファイル未生成")
def test_frozen_sigma_file_matches_the_current_channel_spec():
    """リポジトリに凍結してある σ_c が、現在のチャンネル定義で測られたものであること。"""
    loaded = CALIB.load_sigma(None)
    assert loaded["status"] == "loaded"
    assert loaded["doc"]["meta"]["channel_spec_sha256"] == CH.spec_sha256(), \
        "チャンネル定義が変わっている: scripts/measure_sigma.py で測り直すこと"
    assert CALIB.sigma_of(loaded), "凍結ファイルに usable なチャンネルが無い"


# --------------------------------------------------------------------------- #
# (H) 宣言(検収基準 5)
# --------------------------------------------------------------------------- #
def test_feature_is_declared_in_the_registry():
    feature = {f.id: f for f in R.FEATURES}.get("cognition.channels.enabled")
    assert feature is not None, "機能レジストリに宣言が無い"
    assert feature.repro_tier == "strict", "観測専用・乱数ゼロなので strict"
    assert feature.affects_k is False, "LLM 呼び出し点を増減しない"
    assert feature.fingerprint_risk == "none", "プロンプトを 1 バイトも変えない"


def test_new_conf_keys_are_classified_in_timeconv():
    assert T.covers("cognition.channels.every_steps"), \
        "Δt 分類テーブルに新キーの宣言が無い"
    assert T.classify("cognition.channels.every_steps")[0] == T.INVARIANT


def test_verify_mode_keeps_channels_available():
    """strict 等級なので run.mode=verify(対照実験)でも自動 OFF にされない。"""
    cfg = load_config(["run.mode=verify", "cognition.channels.enabled=true"])
    report = R.describe(cfg)
    assert bool(cfg.cognition.channels.enabled) is True
    assert "cognition.channels.enabled" in {f["id"] for f in report["enabled"]}
