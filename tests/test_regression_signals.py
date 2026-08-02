"""退行シグナル監視 + 判定スクリプト 第91バッチ(設計 §3)のテスト。

方針(第70 echo / 第75 dunbar / 第87 engaged の鉄則を継承):
- **観測側のみ**: `observer.regression.enabled` を切り替えても **L1 は完全に一致**する
  (変わるのは L2 の列だけ)。プロンプト 1 バイト不変・LLM 呼数不変・乱数ゼロ。
- 既定 OFF = **列が 1 つも生えない**・state 不在・manifest/summary にキーなし。
- 値は**手計算**および **L1 からの独立再計算**と一致させる(空回り防止)。
- 第87 の申し送り: 定型応答(engaged TEMPLATE)が語彙系から除外され、件数が列に残る。
- resume: `_regression_state` が checkpoint round-trip で復元され L2 が straight と一致。
- 判定スクリプト(scripts/detect_regression.py)は Mann-Kendall / Theil-Sen / 間引き /
  分散崩壊診断を手計算・合成系列で固定する。閾値が**引数必須**であることも固定する。

検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import registry as R
from society.cognition import engaged as EG
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import regression as RG
from society.observer.schema import EVENT_KINDS

REPO_ROOT = Path(__file__).resolve().parents[1]

ON = {"observer.regression.enabled": "true"}
FIRE_ON = {"cognition.fire.enabled": "true"}
BASE_COLS = RG.BASE_COLUMNS
FIRE_COLS = RG.FIRE_COLUMNS


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name, steps, n, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n=12, steps=24, **ov):
    return Simulation(_cfg(name, steps, n, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _l2_cols(sim):
    return pq.read_table(sim.out_dir / "l2_metrics.parquet").column_names


def _detect_module():
    """scripts/detect_regression.py を module として読み込む(CLI を起動せずに検査する)。"""
    path = REPO_ROOT / "scripts" / "detect_regression.py"
    spec = importlib.util.spec_from_file_location("_detect_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DR = _detect_module()


# --------------------------------------------------------------------------- #
# 1) 純ヘルパ(手計算一致)
# --------------------------------------------------------------------------- #
def test_tokens_are_char_ngrams_with_symbols_dropped():
    assert RG.tokens("あいうえ", 2) == ("あい", "いう", "うえ")
    assert RG.tokens("あ、い。う", 2) == ("あい", "いう")     # 記号は落ちる
    assert RG.tokens("あ", 2) == ("あ",)                      # n 未満は全体 1 要素
    assert RG.tokens("", 2) == ()
    assert RG.tokens("、。 ", 2) == ()                        # 記号だけなら空


def test_entropy_bits_hand_computed():
    assert RG.entropy_bits({}) == 0.0
    assert RG.entropy_bits({"a": 5}) == 0.0                   # 1 種類 = 0 bit
    assert abs(RG.entropy_bits({"a": 1, "b": 1}) - 1.0) < 1e-12
    assert abs(RG.entropy_bits({"a": 1, "b": 1, "c": 1, "d": 1}) - 2.0) < 1e-12
    # 1/2, 1/4, 1/4 → 1.5 bit
    assert abs(RG.entropy_bits({"a": 2, "b": 1, "c": 1}) - 1.5) < 1e-12


def test_quantile_linear_interpolation_matches_numpy_definition():
    xs = [0.0, 1.0, 2.0, 3.0]
    assert RG.quantile(xs, 0.0) == 0.0
    assert RG.quantile(xs, 1.0) == 3.0
    assert abs(RG.quantile(xs, 0.5) - 1.5) < 1e-12
    assert abs(RG.quantile(xs, 0.10) - 0.3) < 1e-12
    assert RG.quantile([], 0.5) == 0.0
    assert RG.quantile([7.0], 0.9) == 7.0


def test_unbiased_var_is_n_minus_one():
    # 標本 [0,2] → 平均1・偏差平方和2 → 不偏 2/1 = 2.0(母分散なら 1.0)
    assert RG.unbiased_var([0.0, 2.0]) == 2.0
    assert RG.unbiased_var([5.0]) == 0.0
    assert RG.unbiased_var([]) == 0.0
    assert RG.unbiased_var([3.0, 3.0, 3.0]) == 0.0


def test_act_dispersion_hand_computed():
    """2 個体が完全に別の act だけをする → 各次元の不偏分散 = 0.5、次元平均 = 1/D。"""
    i_arrive = RG.ACT_KINDS.index("arrive")
    i_speak = RG.ACT_KINDS.index("speak")
    acts = {1: {i_arrive: 4}, 2: {i_speak: 4}}
    var, ent, n = RG.act_dispersion(acts)
    assert n == 2
    assert ent == 0.0                                    # 個体内は 1 種類だけ = 0 bit
    # p ベクトルは (1,0,…) と (0,…,1,…) → 2 次元だけ Var=0.5、他は 0
    assert abs(var - (2 * 0.5) / len(RG.ACT_KINDS)) < 1e-12
    # 完全に同じ行動をする 2 個体 → 個体間分散 0・個体内エントロピー 1 bit
    same = {1: {i_arrive: 2, i_speak: 2}, 2: {i_arrive: 3, i_speak: 3}}
    var2, ent2, n2 = RG.act_dispersion(same)
    assert (var2, n2) == (0.0, 2)
    assert abs(ent2 - 1.0) < 1e-12


def test_act_kinds_are_all_registered_event_kinds():
    """ACT_KINDS の綴りが L1 のイベント種と一致する(typo で永久に 0 になる事故の防止)。"""
    unknown = [k for k in RG.ACT_KINDS if k not in EVENT_KINDS]
    assert unknown == [], f"未登録のイベント種を act に入れている: {unknown}"
    # 移動の内訳は 1 行動が step 数ぶん増殖するので**入れない**という設計の固定
    for k in ("move_segment", "route_start", "cog_fire", "cog_event"):
        assert k not in RG.ACT_KINDS


def test_direction_table_covers_exactly_the_judgeable_columns():
    """判定方向は「上下に意味がある列」だけに付く(分母・母数の列には付けない)。"""
    for col in RG.REGRESSION_DIRECTION:
        assert col in RG.COLUMNS
        assert RG.REGRESSION_DIRECTION[col] in (-1, 1)
    for col in ("reg_act_agents", "reg_visit_nodes", "reg_vocab_tokens",
                "reg_vocab_excluded", "reg_fire_rate_p10", "reg_fire_rate_p50",
                "reg_fire_rate_p90"):
        assert col not in RG.REGRESSION_DIRECTION
    # §3 が名指しする 4 群がすべて方向表にある
    for col in ("reg_act_between_var", "reg_visit_entropy", "reg_vocab_entropy",
                "reg_ngram_repeat_rate", "reg_fire_zero_frac",
                "reg_fire_sat_frac"):
        assert col in RG.REGRESSION_DIRECTION


# --------------------------------------------------------------------------- #
# 2) 既定 OFF = 完全に無風
# --------------------------------------------------------------------------- #
def test_default_off_leaves_l1_identical_and_no_columns(tmp_path):
    base = _sim(tmp_path, "base", steps=48)
    base.run()
    off = _sim(tmp_path, "off", steps=48, **{"observer.regression.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off)
    cols = _l2_cols(base)
    assert [c for c in cols if c.startswith("reg_")] == []
    assert getattr(base, "_regression_state", None) is None
    assert RG.scalars(base) is None
    assert RG.provenance(base) is None


def test_default_off_has_no_manifest_or_summary_key(tmp_path):
    sim = _sim(tmp_path, "prov_off", steps=12)
    sim.run()
    man = json.loads((sim.out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summ = json.loads((sim.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "regression" not in man
    assert "regression" not in summ


def test_on_does_not_change_the_world_at_all(tmp_path):
    """★観測が世界を変えない: ON/OFF で L1 バイト一致・LLM 呼数一致。"""
    off = _sim(tmp_path, "w_off", steps=48)
    off.run()
    on = _sim(tmp_path, "w_on", steps=48, **ON)
    on.run()
    assert _l1(off) == _l1(on), "regression ON で L1 が動いた(観測がシムを変えている)"
    assert off.llm.calls == on.llm.calls


def test_registry_declares_the_toggle():
    assert "observer.regression.enabled" in R.BY_ID
    row = R.BY_ID["observer.regression.enabled"]
    assert row.repro_tier == "strict"           # 読むだけ = 決定論
    assert row.affects_k is False               # LLM 呼び出し点を 1 つも増減しない
    assert row.fingerprint_risk == "none"       # プロンプトに 1 バイトも出ない


# --------------------------------------------------------------------------- #
# 3) ON の列構成
# --------------------------------------------------------------------------- #
def test_on_without_fire_emits_base_columns_only(tmp_path):
    sim = _sim(tmp_path, "nofire", steps=24, **ON)
    sim.run()
    cols = _l2_cols(sim)
    for c in BASE_COLS:
        assert c in cols, f"{c} が無い"
    for c in FIRE_COLS:
        assert c not in cols, f"fire OFF なのに {c} が出ている"


def test_on_with_fire_emits_all_columns(tmp_path):
    sim = _sim(tmp_path, "withfire", steps=24, **ON, **FIRE_ON)
    sim.run()
    cols = _l2_cols(sim)
    for c in RG.COLUMNS:
        assert c in cols, f"{c} が無い"
    prov = json.loads(
        (sim.out_dir / "run_manifest.json").read_text(encoding="utf-8"))["regression"]
    assert prov["fire_columns"] is True
    assert prov["columns"] == list(RG.COLUMNS)


def test_same_seed_two_runs_are_identical(tmp_path):
    a = _sim(tmp_path, "det_a", steps=36, **ON, **FIRE_ON)
    a.run()
    b = _sim(tmp_path, "det_b", steps=36, **ON, **FIRE_ON)
    b.run()
    ta = pq.read_table(a.out_dir / "l2_metrics.parquet").to_pylist()
    tb = pq.read_table(b.out_dir / "l2_metrics.parquet").to_pylist()
    assert ta == tb


# --------------------------------------------------------------------------- #
# 4) 独立検算(L1 から素で数え直した値と一致する)
# --------------------------------------------------------------------------- #
def _recompute_from_l1(sim, window: int, ngram: int, drop_texts, with_fire: bool):
    """L2 と**別実装**で最終行の窓を数え直す(素の Python ループ。共有コードを使わない)。"""
    events = [e for e in sim.logger.events]
    last = max(int(e.step) for e in events)
    floor = last - window + 1
    acts: dict = {}
    visits: dict = {}
    toks: dict = {}
    tok_n = excl = 0
    fires: dict = {}
    active: dict = {}
    for e in events:
        if int(e.step) < floor:
            continue
        aid = int(e.agent_id)
        if aid >= 0:
            active[aid] = active.get(aid, 0) + 1
        if e.kind in RG.ACT_KINDS and aid >= 0:
            row = acts.setdefault(aid, {})
            j = RG.ACT_KINDS.index(e.kind)
            row[j] = row.get(j, 0) + 1
        if e.kind == "arrive":
            node = e.payload.get("node")
            if node is not None:
                visits[str(node)] = visits.get(str(node), 0) + 1
        if e.kind in RG.UTTERANCE_KINDS:
            text = e.payload.get("text")
            if isinstance(text, str) and text.strip():
                txt = text.strip()
                if txt in drop_texts:
                    excl += 1
                else:
                    s = "".join(ch for ch in txt if ch not in RG._DROP)
                    grams = ([s] if 0 < len(s) < ngram
                             else [s[i:i + ngram] for i in range(len(s) - ngram + 1)])
                    for g in grams:
                        toks[g] = toks.get(g, 0) + 1
                    tok_n += len(grams)
        if e.kind == "cog_fire" and aid >= 0:
            fires[aid] = fires.get(aid, 0) + 1
    # 個体間分散(不偏)を素で計算
    rows = []
    for aid in sorted(acts):
        total = sum(acts[aid].values())
        vec = [0.0] * len(RG.ACT_KINDS)
        for j, c in acts[aid].items():
            vec[j] = c / total
        rows.append(vec)
    n = len(rows)
    dim_sum = 0.0
    for j in range(len(RG.ACT_KINDS)):
        col = [r[j] for r in rows]
        if n >= 2:
            mean = sum(col) / n
            dim_sum += sum((x - mean) ** 2 for x in col) / (n - 1)
    out = {
        "reg_act_between_var": round(dim_sum / len(RG.ACT_KINDS), 9),
        "reg_act_agents": n,
        "reg_visit_nodes": len(visits),
        "reg_vocab_tokens": tok_n,
        "reg_vocab_excluded": excl,
        "reg_ngram_repeat_rate": (round(1.0 - len(toks) / tok_n, 6) if tok_n else 0.0),
    }
    if with_fire:
        span = min(window, last + 1)
        rates = sorted(fires.get(a, 0) / span for a in sorted(active))
        out["reg_fire_zero_frac"] = (round(sum(1 for r in rates if r <= 0) / len(rates), 6)
                                     if rates else 0.0)
    return out


def test_columns_match_independent_recomputation_from_l1(tmp_path):
    """L2 の最終行が、L1 を素で数え直した値と**完全に**一致する。"""
    sim = _sim(tmp_path, "recalc", n=15, steps=60, **ON, **FIRE_ON)
    sim.run()
    row = pq.read_table(sim.out_dir / "l2_metrics.parquet").to_pylist()[-1]
    ref = _recompute_from_l1(sim, 144, 2, (EG.TEMPLATE,), with_fire=True)
    for key, expect in ref.items():
        assert row[key] == expect, f"{key}: L2={row[key]} vs L1 再計算={expect}"
    assert row["reg_act_agents"] > 0, "テスト前提が崩れた(act が 1 件も無い)"


# --------------------------------------------------------------------------- #
# 5) 第87 の申し送り: 定型応答を語彙系から除外する
# --------------------------------------------------------------------------- #
def test_template_replies_are_excluded_from_vocabulary_and_counted(tmp_path):
    """engaged の定型応答は語彙指標の分子に入らず、件数が列と summary に残る。"""
    ov = {**ON, **FIRE_ON, "cognition.engaged.enabled": "true"}
    sim = _sim(tmp_path, "tmpl", n=25, steps=96, **ov)
    sim.run()
    summ = json.loads((sim.out_dir / "summary.json").read_text(encoding="utf-8"))
    n_template = summ["engaged"]["template_replies"]
    assert n_template > 0, "テスト前提が崩れた(定型応答が 1 件も出ていない)"
    assert summ["regression"]["excluded_total"] == n_template, \
        "除外総数が engaged の定型応答件数と一致しない"
    # 除外された文の n-gram が語彙タリーへ入っていないこと(state を直接見る)
    st = sim._regression_state
    tmpl_grams = RG.tokens(EG.TEMPLATE, 2)
    counted = sum(st["tok_totals"].get(g, 0) for g in tmpl_grams)
    # 定型文と同じ 2-gram が他の発話に現れる可能性はあるので「窓内の定型件数ぶんは
    # 引かれている」ことを、除外 OFF ランとの差で示す
    off = _sim(tmp_path, "tmpl_off", n=25, steps=96,
               **{**ov, "observer.regression.exclude_template": "false"})
    off.run()
    st_off = off._regression_state
    counted_off = sum(st_off["tok_totals"].get(g, 0) for g in tmpl_grams)
    assert counted_off > counted, "exclude_template=false でも語彙タリーが同じ(除外が効いていない)"
    assert off._regression_state["excl_n"] == 0


def test_excluded_texts_respects_the_switch():
    assert RG.excluded_texts({"exclude_template": True}) == (EG.TEMPLATE,)
    assert RG.excluded_texts({"exclude_template": False}) == ()


# --------------------------------------------------------------------------- #
# 6) 窓・resume
# --------------------------------------------------------------------------- #
def test_rolling_window_drops_old_events():
    """窓幅を超えた step のイベントはタリーから落ちる(有界メモリの担保)。"""
    class _E:
        def __init__(self, step, aid, kind, **payload):
            self.step, self.agent_id, self.kind, self.payload = step, aid, kind, payload

    cfg = dict(RG._DEFAULTS)
    cfg["window_steps"] = 3
    st = RG._fresh_state()
    RG._ingest(st, [_E(0, 1, "arrive", node="A"), _E(1, 1, "arrive", node="B")], cfg)
    assert len(st["visit_totals"]) == 2
    RG._ingest(st, [_E(5, 1, "arrive", node="C")], cfg)   # 窓 [3..5] に押し出される
    assert dict(st["visit_totals"]) == {"C": 1}
    assert len(st["buckets"]) == 1


def test_resume_matches_straight(tmp_path):
    """mid-day resume が straight と L1/L2/L3 一致(_regression_state の中央管理)。"""
    ov = {**ON, **FIRE_ON, "observer.checkpoint_every": 48}
    split, total, n = 48, 96, 15
    straight_dir = tmp_path / "rg_straight"
    straight = Simulation(_cfg("rg_straight", total, n, **ON, **FIRE_ON),
                          out_dir=straight_dir)
    straight.run()

    d = tmp_path / "rg_resumed"
    sim1 = Simulation(_cfg("rg_resumed", split, n, **ov), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    assert sim1._regression_state["buckets"], "分割点までに窓が空(検証にならない)"
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("rg_resumed", total, n, **ov), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(regression resume)"


def test_checkpoint_round_trip_restores_the_window(tmp_path):
    sim = _sim(tmp_path, "ck", n=12, steps=1, **ON)
    for step in range(12):
        scheduler.run_step(sim, step)
    RG.scalars(sim)
    before = RG.finalize(sim._regression_state, RG.cfg_of(sim), False)
    path = tmp_path / "ck" / "checkpoint" / "ckpt-000012.pkl.gz"
    checkpoint.save(sim, 12, path)
    sim2 = _sim(tmp_path, "ck2", n=12, steps=1, **ON)
    checkpoint.load(sim2, path)
    assert sim2._regression_processed == 0
    assert RG.finalize(sim2._regression_state, RG.cfg_of(sim2), False) == before


# --------------------------------------------------------------------------- #
# 7) 判定スクリプト: Mann-Kendall / Theil-Sen / 間引き
# --------------------------------------------------------------------------- #
def test_mann_kendall_on_strictly_increasing_series():
    mk = DR.mann_kendall([1, 2, 3, 4, 5, 6, 7, 8])
    assert mk["S"] == 28                              # n(n-1)/2 = 28 の全ペアが正
    assert mk["z"] > 0 and mk["p"] < 0.01
    assert abs(mk["sen_slope"] - 1.0) < 1e-12


def test_mann_kendall_on_strictly_decreasing_and_flat_series():
    dec = DR.mann_kendall([8, 7, 6, 5, 4, 3, 2, 1])
    assert dec["S"] == -28 and dec["z"] < 0 and dec["p"] < 0.01
    assert abs(dec["sen_slope"] + 1.0) < 1e-12
    flat = DR.mann_kendall([5, 5, 5, 5, 5, 5])
    assert flat["S"] == 0 and flat["p"] == 1.0 and flat["sen_slope"] == 0.0


def test_mann_kendall_skips_short_series():
    mk = DR.mann_kendall([1, 2, 3])
    assert mk["method"].startswith("skipped") and mk["p"] == 1.0


def test_downsample_applies_warmup_and_stride():
    steps = list(range(0, 20))
    vals = [float(s) for s in steps]
    s, v = DR.downsample(steps, vals, stride=5, warmup=0)
    assert s == [0, 5, 10, 15] and v == [0.0, 5.0, 10.0, 15.0]
    s2, v2 = DR.downsample(steps, vals, stride=5, warmup=6)
    assert s2 == [6, 11, 16] and v2 == [6.0, 11.0, 16.0]
    # None は落ちる(列が無いランを跨いでも壊れない)
    s3, _ = DR.downsample(steps, [None] * 20, stride=5, warmup=0)
    assert s3 == []


def test_loglog_slope_hand_computed():
    pts = [{"n": 10, "var": 1.0}, {"n": 100, "var": 0.1}, {"n": 1000, "var": 0.01}]
    fit = DR.loglog_slope(pts)
    assert abs(fit["b"] + 1.0) < 1e-9                 # 10 倍で 1/10 → b = -1
    flat = DR.loglog_slope([{"n": 10, "var": 2.0}, {"n": 100, "var": 2.0}])
    assert abs(flat["b"]) < 1e-9
    assert DR.loglog_slope([{"n": 10, "var": 1.0}])["b"] is None


def test_subsample_band_is_flat_for_an_exchangeable_population():
    """★診断の根拠: 同一母集団からの部分抽出では不偏分散の期待値が m に依存しない。"""
    i_a, i_b = RG.ACT_KINDS.index("arrive"), RG.ACT_KINDS.index("speak")
    # 個体ごとに arrive:speak の比を変える(= 個体間分散がある交換可能な母集団)
    acts = {}
    for aid in range(200):
        k = aid % 10
        acts[aid] = {i_a: 1 + k, i_b: 11 - k}
    band = DR.subsample_band(acts, [10, 25, 50, 100, 200], reps=60, seed=0)
    means = [b["mean"] for b in band]
    assert len(band) == 5
    ref = means[-1]
    for m in means:
        assert abs(m - ref) / ref < 0.25, f"部分抽出の期待値が m に依存した: {means}"
    # 帯の幅は m とともに縮む
    widths = [b["p95"] - b["p05"] for b in band]
    assert widths[0] > widths[-1]


# --------------------------------------------------------------------------- #
# 8) 判定スクリプト: エンドツーエンド
# --------------------------------------------------------------------------- #
def _write_fake_run(dirpath: Path, series: dict, n_rows: int, window: int = 4):
    """L2 だけを持つ最小のラン dir を作る(合成系列で判定式そのものを固定する)。"""
    import pyarrow as pa
    dirpath.mkdir(parents=True, exist_ok=True)
    cols = {"step": list(range(n_rows))}
    cols.update({k: list(v) for k, v in series.items()})
    pq.write_table(pa.table(cols), dirpath / "l2_metrics.parquet")
    (dirpath / "run_manifest.json").write_text(
        json.dumps({"regression": {"window_steps": window}}), encoding="utf-8")
    return dirpath


def test_end_to_end_flags_a_collapsing_vocabulary(tmp_path):
    n = 40
    run = _write_fake_run(tmp_path / "bad", {
        "reg_vocab_entropy": [10.0 - 0.2 * i for i in range(n)],   # 単調低下
        "reg_ngram_repeat_rate": [0.5 + 0.005 * i for i in range(n)],  # 単調上昇
        "reg_act_between_var": [0.01] * n,                          # 平坦
    }, n, window=4)
    res = DR.trend_report(run, alpha=0.05, min_rel_slope=0.02, stride=0, warmup=0)
    assert res["verdict"] == "REGRESSION"
    assert set(res["flagged"]) == {"reg_vocab_entropy", "reg_ngram_repeat_rate"}
    assert res["signals"]["reg_act_between_var"]["verdict"] == "OK"
    assert res["stride"] == 4 and res["stride_source"] == "run(window_steps)"


def test_end_to_end_passes_a_healthy_run(tmp_path):
    n = 40
    # 上下に揺れるが単調ではない系列 = 退行なし
    vals = [8.0 + (1.0 if i % 2 else -1.0) for i in range(n)]
    run = _write_fake_run(tmp_path / "good",
                          {"reg_vocab_entropy": vals}, n, window=4)
    res = DR.trend_report(run, alpha=0.05, min_rel_slope=0.02, stride=0, warmup=0)
    assert res["verdict"] == "OK"
    assert res["flagged"] == []


def test_wrong_direction_is_not_flagged(tmp_path):
    """語彙エントロピーが**上がる**のは退行ではない(方向の連言が効いている)。"""
    n = 40
    run = _write_fake_run(tmp_path / "up",
                          {"reg_vocab_entropy": [1.0 + 0.2 * i for i in range(n)]},
                          n, window=4)
    res = DR.trend_report(run, alpha=0.05, min_rel_slope=0.02, stride=0, warmup=0)
    assert res["signals"]["reg_vocab_entropy"]["verdict"] == "OK"
    assert res["verdict"] == "OK"


def test_effect_size_gate_blocks_a_tiny_but_significant_trend(tmp_path):
    """p は有意でも相対傾きが閾値未満なら退行としない(n 依存の過検出を止める)。"""
    n = 60
    run = _write_fake_run(tmp_path / "tiny",
                          {"reg_vocab_entropy": [10.0 - 1e-4 * i for i in range(n)]},
                          n, window=1)
    res = DR.trend_report(run, alpha=0.05, min_rel_slope=0.02, stride=1, warmup=0)
    sig = res["signals"]["reg_vocab_entropy"]
    assert sig["p"] < 0.05, "テスト前提が崩れた(単調系列なのに有意でない)"
    assert abs(sig["rel_slope"]) < 0.02
    assert sig["verdict"] == "OK"


def test_cli_requires_thresholds():
    """閾値の既定値をコードに埋め込んでいない(引数必須)。"""
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "detect_regression.py"), "."],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "--alpha" in out.stderr and "--min-rel-slope" in out.stderr


@pytest.mark.xdist_group("subprocess_heavy")
def test_cli_quick_mode_emits_one_json_line_and_never_fails(tmp_path):
    """watchdog から呼ぶ経路: 1 行 JSON・ファイルを書かない・**必ず exit 0**。"""
    n = 40
    run = _write_fake_run(tmp_path / "quick",
                          {"reg_vocab_entropy": [10.0 - 0.2 * i for i in range(n)]},
                          n, window=4)
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "detect_regression.py"),
         str(run), "--quick", "--alpha", "0.05", "--min-rel-slope", "0.02"],
        capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    blob = json.loads(lines[0])
    assert blob["verdict"] == "REGRESSION"
    assert "reg_vocab_entropy" in blob["flagged"]
    assert not (run / "regression_report.md").exists()
    assert not (run / "regression_report.json").exists()


def test_reads_in_progress_parts_and_skips_the_unfinished_one(tmp_path):
    """実行中のランを覗く経路: **完結した part だけ**を index 順に連結する。

    ★第77 の教訓(読んでいる最中の part をランが unlink できずシムが finalize で落ちる)を
      避けるため、開くのは live_viewer._open_shared 経由。ここではその配線と
      「書きかけは読まない」規律を固定する。
    """
    import pyarrow as pa
    run = tmp_path / "live"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"regression": {"window_steps": 4}}), encoding="utf-8")
    pq.write_table(pa.table({"step": [0, 1], "reg_vocab_entropy": [9.0, 8.5]}),
                   run / "l2_metrics.part-000000.parquet")
    pq.write_table(pa.table({"step": [2, 3], "reg_vocab_entropy": [8.0, 7.5]}),
                   run / "l2_metrics.part-000001.parquet")
    # part-000002 は書きかけ(末尾 magic が無い)= 読まない
    (run / "l2_metrics.part-000002.parquet").write_bytes(b"PAR1" + b"\x00" * 64)

    steps, cols, meta = DR.load_l2(run)
    assert meta["source"] == "parts" and meta["parts"] == 2
    assert meta["skipped_parts"] == ["l2_metrics.part-000002.parquet"]
    assert steps == [0, 1, 2, 3]
    assert cols["reg_vocab_entropy"] == [9.0, 8.5, 8.0, 7.5]
    assert DR.run_window_steps(run) == 4


def test_run_window_steps_falls_back_to_config_yaml(tmp_path):
    """manifest が無いラン(古い/最中)では config.yaml の regression ブロックを読む。"""
    run = tmp_path / "cfgonly"
    run.mkdir()
    (run / "config.yaml").write_text(
        "observer:\n  regression:\n    enabled: true\n    window_steps: 72\n",
        encoding="utf-8")
    assert DR.run_window_steps(run) == 72
    assert DR.run_window_steps(tmp_path / "nothing_here") == 144   # 最終退避=1 日


def test_variance_svg_is_wellformed_and_marks_outliers():
    vc = {
        "points": [{"run": "a", "n": 50, "var": 0.02, "inside_band": True},
                   {"run": "b", "n": 200, "var": 0.001, "inside_band": False}],
        "band": [{"m": 50, "mean": 0.02, "p05": 0.015, "p95": 0.026},
                 {"m": 200, "mean": 0.02, "p05": 0.018, "p95": 0.022}],
        "band_from": "b", "reps": 100, "fit": {"b": -1.0},
    }
    svg = DR.variance_svg(vc)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<circle") == 2
    assert "#e6550d" in svg                       # 帯の外の点は色を変える
    assert "1/√N" in svg                          # 参照傾きの凡例


def test_variance_collapse_detects_homogenization(tmp_path, monkeypatch):
    """★N が大きいほど均質 = 小 N の実測が帯の**上**に出る → 崩壊を疑う。

    帯は最大 N のランから同数を抽出した帰無分布なので、「大きいランのほうが均質」
    のときにだけ小 N の実測が帯の上へはみ出す。ここが §3 の切り分けの心臓部。
    """
    i_a, i_b = RG.ACT_KINDS.index("arrive"), RG.ACT_KINDS.index("speak")

    def diverse(n):
        return {aid: {i_a: 1 + (aid % 10), i_b: 11 - (aid % 10)} for aid in range(n)}

    def uniform(n):                    # 全員そっくり = 均質化した母集団
        return {aid: {i_a: 6, i_b: 6} for aid in range(n)}

    # 小 N は多様・大 N は均質 = 「規模を上げたら全員同じ行動になった」世界
    fake = {"small": (diverse(40), 200, 144), "big": (uniform(200), 200, 144)}

    def _fake_counts(run_dir, window_steps, batch_rows=0):
        return fake[Path(run_dir).name]

    monkeypatch.setattr(DR, "act_counts_from_l1", _fake_counts)
    monkeypatch.setattr(DR, "run_window_steps", lambda p: 144)
    res = DR.variance_collapse([tmp_path / "small", tmp_path / "big"],
                               144, 60, None)
    assert res["verdict"] == "MODEL_HOMOGENIZATION_SUSPECTED"
    assert "small" in res["above_band"]
    assert res["below_band"] == []


def test_variance_collapse_calls_the_opposite_direction_a_population_mismatch(
        tmp_path, monkeypatch):
    """帯の**下**は崩壊ではない(N が大きいほど多様)= 母集団構成の不一致を疑う。

    実測(平坦名簿を巡回複製した N=50/100/200 の mock ラン)で実際に起きた形。
    """
    i_a, i_b = RG.ACT_KINDS.index("arrive"), RG.ACT_KINDS.index("speak")

    def narrow(n, k_mod):              # 小さいほどペルソナ空間が狭い
        return {aid: {i_a: 1 + (aid % k_mod), i_b: 11 - (aid % k_mod)}
                for aid in range(n)}

    fake = {"small": (narrow(40, 2), 200, 144), "big": (narrow(200, 10), 200, 144)}
    monkeypatch.setattr(DR, "act_counts_from_l1",
                        lambda rd, w, batch_rows=0: fake[Path(rd).name])
    monkeypatch.setattr(DR, "run_window_steps", lambda p: 144)
    res = DR.variance_collapse([tmp_path / "small", tmp_path / "big"],
                               144, 60, None)
    assert res["verdict"] == "POPULATION_MISMATCH_SUSPECTED"
    assert "small" in res["below_band"]
    assert res["above_band"] == []


def test_variance_collapse_says_no_collapse_for_exchangeable_runs(tmp_path,
                                                                 monkeypatch):
    i_a, i_b = RG.ACT_KINDS.index("arrive"), RG.ACT_KINDS.index("speak")

    def diverse(n):
        return {aid: {i_a: 1 + (aid % 10), i_b: 11 - (aid % 10)} for aid in range(n)}

    fake = {"n50": (diverse(50), 200, 144), "n200": (diverse(200), 200, 144)}
    monkeypatch.setattr(DR, "act_counts_from_l1",
                        lambda rd, w, batch_rows=0: fake[Path(rd).name])
    monkeypatch.setattr(DR, "run_window_steps", lambda p: 144)
    res = DR.variance_collapse([tmp_path / "n50", tmp_path / "n200"], 144, 60, None)
    assert res["verdict"] == "NO_COLLAPSE"
    assert res["below_band"] == []
    assert abs(res["fit"]["b"]) < 0.25


def test_variance_collapse_reads_real_l1(tmp_path):
    """act_counts_from_l1 が実ランの L1 から L2 と同じ n を出す(配線の実証)。"""
    sim = _sim(tmp_path, "vc_real", n=14, steps=48, **ON)
    sim.run()
    acts, last_step, span = DR.act_counts_from_l1(sim.out_dir, 144)
    var, ent, n = RG.act_dispersion(acts)
    row = pq.read_table(sim.out_dir / "l2_metrics.parquet").to_pylist()[-1]
    assert last_step == 47
    assert n == row["reg_act_agents"]
    assert round(var, 9) == row["reg_act_between_var"]
    assert round(ent, 6) == row["reg_act_entropy_mean"]


def test_report_markdown_contains_the_required_sections(tmp_path):
    n = 40
    run = _write_fake_run(tmp_path / "md",
                          {"reg_vocab_entropy": [10.0 - 0.2 * i for i in range(n)]},
                          n, window=4)
    res = {"schema": 1, "run_dir": str(run),
           "thresholds": {"alpha": 0.05, "min_rel_slope": 0.02},
           "trend": DR.trend_report(run, 0.05, 0.02, 0, 0),
           "variance_collapse": None}
    md = DR.render(res)
    assert "§A 時系列トレンド検定" in md
    assert "§C 正直な限界" in md
    assert "--alpha 0.05" in md
    assert "REGRESSION" in md
