"""IF-D(痕跡 = 場所のイベント履歴 `world.traces`)のテスト。

正典
  - docs/research/if-lane-research.md **§1**(スティグマジー。Grassé 1959 → Heylighen
    2011/2016 → Parunak。★**演算は集約と蒸発の 2 つだけ**= propagation factor 0 /
    ★減衰は欠陥ではなく**機能**「古い痕跡は無関係ではなく**誤誘導**になる」/
    ★TTL は 3 階層(transient / daily=既存 flyer 144 step / persistent)/
    ★既存の貼り紙は完全なスティグマジー実装で、本機構はその一般化)
  - docs/research/llm-world-interface-audit.md **§3**(痕跡カテゴリの現状 = 貼り紙 /
    出店 / 場所ラベルの個別 3 種のみ。場所のイベント履歴は L1 に座標付きで残るが
    **シム内から読む経路がゼロ**)
  - docs/plans/if-sv-p4-plan.md 「IF-D」行

守るもの(検収基準の順)
  ① OFF(既定)= ゴールデン L1 バイト一致・キー不発生・state も属性も生えない
  ② ON: 対象イベント → trace_mark → **後から来た者**の材料に trace_line 1 行 /
        強度が閾値未満では出ない / 当事者(同席者)には出ない
  ③ 蒸発: 3 階層の半減期どおり減衰・**日境界 1 回**・persistent は減らない
  ④ 最強 1 件のみ・定型文に数字/実験条件なし(rumors の 2 段検査を流用)
  ⑤ ON 同 seed 2 ラン一致
  ⑥ resume == straight
  ⑦ LLM 呼数 ON/OFF 完全一致 + プロンプト差分が trace_line 1 欄に閉じる
  ⑧ 契約列挙(perception_contract)の追随
  ⑨ ★propagation がコードとして存在しない(AST の静的検査)
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

from society import registry as R
from society import traces as T
from society.cognition import deliberate
from society.cognition import perception_contract as PC
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

# test_rumors.py:45 / test_rejection_notify.py:44 と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

OFF = {"world.traces.enabled": "false"}
ON = {"world.traces.enabled": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_rumors.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _l1x(sim):
    return [[e.step, e.agent_id, e.kind, e.llm_call_id,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


class _FixedLLM:
    """**プロンプト非依存**の巡回応答スタブ(test_rumors / test_rejection_notify と同型)。

    mock は `hub.stream("mock", rng_key, prompt)` = プロンプト全文で乱数を引くので、
    1 行増えるだけで応答が変わり世界が分岐して呼数の比較にならない。応答を
    **呼び出し番号の関数**にすることで「1 行増えたこと」が行動列を変えない世界を作る。
    """

    name = "fixed"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.hits = 0
        self.prompts: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        self.prompts.append(prompt)
        return out, str(self.calls), False


def _acts(*objs) -> list[str]:
    return [json.dumps(o, ensure_ascii=False) for o in objs]


# 源イベント(host_event)と会話(speak)の両方が立つ巡回応答。
_HOST_AND_TALK = _acts(
    {"action": "host_event", "title": "集まり", "hours_later": 1},
    {"action": "speak", "text": "こんにちは"},
    {"action": "speak", "text": "そうですね"},
    {"action": "speak", "text": "また今度"},
)


def _colocate(dst, src) -> None:
    """dst を src と同席させる(hearers_of の context + 半径を満たす)。"""
    dst.x, dst.y = src.x, src.y
    dst.node = src.node
    dst.loc = src.loc
    dst.building = src.building
    dst.floor = src.floor
    dst.sleeping = False


def _isolate(sim, keep) -> None:
    """keep 以外の全員を範囲外へ退避(= 集約時の同席者を意図した者だけにする)。"""
    ids = {int(a.id) for a in keep}
    for a in sim.agents:
        if int(a.id) not in ids:
            a.loc, a.building = "outside", ""


def _host(sim, agent, step=0, sim_min=0, title="集まり"):
    """実コード(tools._host_event)で source イベント(event_host)を 1 件起こす。"""
    sim.tools._host_event(sim, agent, {"title": title, "hours_later": 1},
                          step, sim_min)


def _open(sim, agent, step=0, sim_min=0, name="屋台"):
    """実コード(tools._open_venture)で source イベント(venture_open)を 1 件起こす。"""
    agent.money = 1_000_000.0
    sim.tools._open_venture(sim, agent, {"name": name, "offer": "たこ焼き"},
                            step, sim_min)


# =========================================================================== #
# (A) 出荷既定・宣言(検収基準 ①の前段)
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.world.traces.enabled) is False
    assert list(cfg.world.traces.kinds) == list(T.DEFAULT_KINDS)
    assert dict(cfg.world.traces.half_life_steps) == {"transient": 18,
                                                      "daily": 144,
                                                      "persistent": 0}


def test_registry_and_schema_declared():
    feat = {f.id: f for f in R.FEATURES}["world.traces.enabled"]
    assert feat.repro_tier == "strict"
    assert feat.affects_k is False        # generate() の呼び出しサイトを 1 つも足さない
    assert feat.fingerprint_risk == "possible"
    assert feat.off_value is False
    assert "trace_mark" in EVENT_KINDS
    # 閲覧イベントは**出さない**(受動観測で毎 step 大量発生するため)
    assert "trace_view" not in EVENT_KINDS


def test_spec_table_is_finite_and_sources_exist():
    """痕跡種の表が有限で、源 L1 種が実在し、階層が 3 つのどれかである。"""
    assert set(T.DEFAULT_KINDS) == set(T.TRACE_SPECS)
    assert T.TIERS == ("transient", "daily", "persistent")
    seen_src: set[str] = set()
    for kind, (srcs, tier, word) in T.TRACE_SPECS.items():
        assert srcs, kind
        assert tier in T.TIERS, kind
        assert word.strip(), kind
        for src in srcs:
            # 綴り間違いで永久に 0 件になるのを防ぐ(rumors と同じ機械固定)
            assert src in EVENT_KINDS, f"源イベント種 {src} が L1 スキーマに無い"
            assert src not in seen_src, f"源 {src} が 2 つの痕跡種に割り当たっている"
            seen_src.add(src)
    # 逆引き表が表と 1 対 1
    assert T.SRC_TO_KIND == {s: k for k, (ss, _t, _w) in T.TRACE_SPECS.items()
                             for s in ss}
    # 3 階層すべてが実際に使われている(単一 TTL に退化していない = 文献の要求)
    assert {T.TRACE_SPECS[k][1] for k in T.TRACE_SPECS} == set(T.TIERS)


def test_unknown_config_values_degrade_to_defaults():
    cfg = T.build_cfg({"enabled": True, "kinds": ["gathering", "存在しない種"],
                       "tier_of": {"gathering": "とんでも", "無い種": "daily"},
                       "half_life_steps": {"daily": -5, "無い階層": 3},
                       "deposit": -1.0})
    assert cfg["kinds"] == ["gathering"]           # 定型語を持たない種は黙って捨てる
    assert cfg["tier_of"] == {}                    # 未知の階層名も未知の種も捨てる
    assert T.tier_of(cfg, "gathering") == T.DAILY  # → 表の既定へ倒れる
    assert cfg["half_life_steps"]["daily"] == 0    # 負値は 0 へ(= 減衰しない)
    assert "無い階層" not in cfg["half_life_steps"]
    assert cfg["deposit"] == 0.0


def test_tier_of_can_be_overridden_by_conf():
    cfg = T.build_cfg({"enabled": True, "tier_of": {"gathering": "transient"}})
    assert T.tier_of(cfg, "gathering") == T.TRANSIENT
    assert T.tier_of(cfg, "opening") == T.PERSISTENT      # 上書きしない種は既定のまま


# --------------------------------------------------------------------------- #
# (A-2) ★静的検査: propagation / LLM / 乱数がコードとして存在しない(検収基準 ⑨)
# --------------------------------------------------------------------------- #
def test_module_has_no_propagation_no_llm_no_rng():
    """★**演算は集約と蒸発の 2 つだけ**(Parunak の propagation factor = 0)。

    散文(docstring / コメント)は対象外 — **実際に評価される識別子**だけを見る。
    """
    tree = ast.parse(Path(T.__file__).read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    funcs = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    idents = attrs | names | funcs
    for bad in ("propagat", "diffus", "spread", "neighbor", "adjacen"):
        hit = sorted(i for i in idents if bad in i.lower())
        assert hit == [], f"拡散に相当する識別子がある(3 演算目を入れてしまった): {hit}"
    # 近傍グラフそのものを触っていない(node は「場所の識別子」としてしか使わない)
    assert "city" not in attrs and "graph" not in attrs and "edges" not in attrs
    # LLM も乱数も引かない
    assert "generate" not in idents and "llm" not in attrs
    assert "stream" not in attrs and "hub" not in attrs and "random" not in attrs


def test_frozen_metric_spec_files_are_untouched():
    """凍結 14 ファイル(metrics_spec.SPEC_FILES)に本バッチの痕跡が 1 つも無い。"""
    from society.observer import metrics_spec as MS
    root = Path(__file__).resolve().parents[1]
    assert len(MS.SPEC_FILES) == 14
    for rel in MS.SPEC_FILES:
        text = (root / rel).read_text(encoding="utf-8")
        assert "traces" not in text and "trace_mark" not in text, \
            f"凍結ファイル {rel} に IF-D の痕跡がある(metrics_spec_hash が動く)"


def test_flyer_implementation_is_untouched():
    """既存の貼り紙(完全なスティグマジー実装)を**書き換えず並置**した。

    flyer = 「意図的な掲示」(marker)/ traces = 「行為の副産物」(trace)で
    Heylighen の区別に対応しており、統合するとその区別が消える。
    """
    text = (Path(__file__).resolve().parents[1] / "src" / "society" / "tools.py"
            ).read_text(encoding="utf-8")
    assert "traces" not in text and "trace_mark" not in text


# =========================================================================== #
# (B) OFF = 現行と 1 バイトも変わらない(検収基準 ①)
# =========================================================================== #
def test_off_matches_golden(tmp_path):
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "tr_golden", **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "IF-D の seam がゴールデンを動かしている"


def test_off_matches_pure_default(tmp_path):
    pure = _sim(tmp_path, "tr_pure")
    pure.run()
    off = _sim(tmp_path, "tr_off", **OFF)
    off.run()
    assert _l1x(pure) == _l1x(off)


def test_off_emits_nothing_and_grows_no_state(tmp_path):
    """OFF では源イベントを起こしても L1 も state も 1 も動かない。"""
    sim = _sim(tmp_path, "tr_off_noop", n_steps=1)
    a = sim.agents[0]
    n_events = len(sim.logger.events)
    _host(sim, a)
    T.phase(sim, 0, 0)
    assert not [e for e in sim.logger.events[n_events:] if e.kind == "trace_mark"]
    assert getattr(sim, "_trace_state", None) is None
    assert T.line_for(sim, a) is None
    assert T.provenance(sim) is None


def test_off_summary_has_no_key(tmp_path):
    sim = _sim(tmp_path, "tr_sum_off", n_steps=24, n_agents=10)
    sim.run()
    summary = json.loads((tmp_path / "tr_sum_off" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert "traces" not in summary


def test_off_material_has_the_key_but_it_is_none(tmp_path):
    """OFF でも契約のキーは存在する(値は None)= 1 行も足さない = バイト一致。"""
    sim = _sim(tmp_path, "tr_off_mat", n_steps=1)
    material = scheduler._gather_material(sim, sim.agents[0], "solo", 0, 0)
    assert "trace_line" in material and material["trace_line"] is None


def test_off_prompts_are_byte_identical(tmp_path):
    def _prompts(name, **ov):
        sim = _sim(tmp_path, name, n_steps=72, n_agents=12, **ov)
        seen: list = []
        inner = sim.llm.generate

        def _gen(prompt, **kw):
            seen.append(prompt)
            return inner(prompt, **kw)
        sim.llm.generate = _gen
        sim.run()
        return seen

    base = _prompts("tr_p_base")
    off = _prompts("tr_p_off", **OFF)
    assert base and base == off


# =========================================================================== #
# (C) ON: 集約 → 後から来た者の 1 行(検収基準 ②)
# =========================================================================== #
def test_event_marks_the_place_and_a_later_visitor_reads_one_line(tmp_path):
    sim = _sim(tmp_path, "tr_mark", n_steps=1, **ON)
    host, later = sim.agents[0], sim.agents[1]
    _isolate(sim, [host])                          # 集約時の同席者は host だけ
    n0 = len(sim.logger.events)
    _host(sim, host)
    T.phase(sim, 0, 0)
    marks = [e for e in sim.logger.events[n0:] if e.kind == "trace_mark"]
    assert len(marks) == 1, f"trace_mark が {len(marks)} 件"
    assert marks[0].payload == {"node": host.node, "kind": "gathering",
                                "tier": "daily"}
    assert marks[0].agent_id == host.id
    assert T.strength_of(sim, host.node, "gathering") == 1.0
    # ---- 後から来た者(集約時に同席していない)は 1 行を読む ----
    _colocate(later, host)
    line = T.line_for(sim, later)
    assert line == T.sentence("gathering") == "この場所では最近、人の集まりがあったようだ。"
    material = scheduler._gather_material(sim, later, "solo", 1, 10)
    assert material["trace_line"] == line
    # ---- 当事者(集約時に同席していた者)には出さない(自分の記憶と二重になる)----
    assert T.line_for(sim, host) is None
    assert scheduler._gather_material(sim, host, "solo", 1, 10)["trace_line"] is None


def test_bystanders_at_the_time_are_excluded_too(tmp_path):
    """同席していた**目撃者**も当事者扱い(hearers_of = 既存の同席者規約)。"""
    sim = _sim(tmp_path, "tr_wit", n_steps=1, **ON)
    host, witness, later = sim.agents[0], sim.agents[1], sim.agents[2]
    _isolate(sim, [host, witness])
    _colocate(witness, host)
    _host(sim, host)
    T.phase(sim, 0, 0)
    rec = sim._trace_state["nodes"][host.node]["gathering"]
    assert set(rec["w"]) == {host.id, witness.id}
    assert T.line_for(sim, witness) is None, "目撃者に伝聞の 1 行が出た"
    _colocate(later, host)
    assert T.line_for(sim, later) == T.sentence("gathering")


def test_below_the_threshold_no_line_is_produced(tmp_path):
    sim = _sim(tmp_path, "tr_thr", n_steps=1, **ON,
               **{"world.traces.line_threshold": 2.0})
    host, later = sim.agents[0], sim.agents[1]
    _isolate(sim, [host])                          # later は集約時ずっと範囲外
    _host(sim, host)
    T.phase(sim, 0, 0)
    assert T.strength_of(sim, host.node, "gathering") == 1.0
    _colocate(later, host)
    assert T.line_for(sim, later) is None, "閾値未満の痕跡が 1 行になった"
    _isolate(sim, [host])                          # 2 件目でも当事者にしない
    _host(sim, host, step=1, sim_min=10)           # 2 件目 → 強度 2.0 = 閾値到達
    T.phase(sim, 1, 10)
    assert T.strength_of(sim, host.node, "gathering") == 2.0
    _colocate(later, host)
    assert T.line_for(sim, later) == T.sentence("gathering")


def test_strength_is_clipped_at_max(tmp_path):
    sim = _sim(tmp_path, "tr_clip", n_steps=1, **ON,
               **{"world.traces.max_strength": 2.0})
    a = sim.agents[0]
    _isolate(sim, [a])
    for i in range(5):
        _host(sim, a, step=i, sim_min=i * 10, title=f"集まり{i}")
        T.phase(sim, i, i * 10)
    assert T.strength_of(sim, a.node, "gathering") == 2.0


def test_kinds_outside_the_conf_list_never_mark(tmp_path):
    sim = _sim(tmp_path, "tr_gate", n_steps=1, **ON,
               **{"world.traces.kinds": "[opening]"})
    a = sim.agents[0]
    _isolate(sim, [a])
    n0 = len(sim.logger.events)
    _host(sim, a)                                  # gathering は kinds に無い
    T.phase(sim, 0, 0)
    assert not [e for e in sim.logger.events[n0:] if e.kind == "trace_mark"]
    assert getattr(sim, "_trace_state")["nodes"] == {}


def test_max_per_step_caps_the_aggregation(tmp_path):
    sim = _sim(tmp_path, "tr_cap", n_steps=1, **ON,
               **{"world.traces.max_per_step": 2})
    a = sim.agents[0]
    _isolate(sim, [a])
    n0 = len(sim.logger.events)
    for i in range(5):
        _host(sim, a, title=f"集まり{i}")
    T.phase(sim, 0, 0)
    marks = [e for e in sim.logger.events[n0:] if e.kind == "trace_mark"]
    assert len(marks) == 2, f"max_per_step が効いていない({len(marks)} 件)"


def test_propagation_off_does_not_gate_the_place_trace(tmp_path):
    """設計判断 (4): 痕跡は**誰の自由文でもない**ので propagation_off では遮断しない。

    同 ablation の契約は「発話の**内容**が他者の文脈に入らない」= **送り手の自由文**の
    遮断で、rumors.on_talk / tools._view_flyers が遮断されるのは他者の書いた文を運ぶから。
    痕跡の 1 行は有限テンプレの純関数なので crowd_line / place_label_line と同じ扱い
    (= 「世界がそうなっている事実」は propagation_off の対象外)。
    """
    sim = _sim(tmp_path, "tr_prop", n_steps=1, **ON,
               **{"ablate.propagation_off": "true"})
    a, later = sim.agents[0], sim.agents[1]
    _isolate(sim, [a])
    _host(sim, a)
    T.phase(sim, 0, 0)
    _colocate(later, a)
    assert T.line_for(sim, later) == T.sentence("gathering")


def test_observation_never_changes_the_trace(tmp_path):
    """★観測(1 行を読むこと)は痕跡を 1 ミリも動かさない(if-lane-research §1-3 示唆 6)。"""
    sim = _sim(tmp_path, "tr_obs", n_steps=1, **ON)
    host, later = sim.agents[0], sim.agents[1]
    _isolate(sim, [host])
    _host(sim, host)
    T.phase(sim, 0, 0)
    _colocate(later, host)
    rec = sim._trace_state["nodes"][host.node]["gathering"]
    before = (float(rec["s"]), dict(rec["w"]), int(rec["step"]))
    n0 = len(sim.logger.events)
    for _ in range(5):
        assert T.line_for(sim, later) == T.sentence("gathering")
    after = (float(rec["s"]), dict(rec["w"]), int(rec["step"]))
    assert after == before, "観測が痕跡(強度・同席者名簿)を書き換えた"
    assert len(sim.logger.events) == n0, "閲覧が L1 にイベントを出した"
    assert sim._trace_state["lines"] == 5           # 数えるのは観測側のタリーだけ


# =========================================================================== #
# (D) 蒸発 = 3 階層の半減期・日境界 1 回(検収基準 ③)
# =========================================================================== #
def _seed(sim, agent, tier_kind="gathering"):
    _isolate(sim, [agent])
    _host(sim, agent)
    T.phase(sim, 0, 0)
    assert T.strength_of(sim, agent.node, tier_kind) == 1.0
    return agent.node


def test_daily_tier_halves_after_one_day(tmp_path):
    """daily = 半減期 144 step(既存 flyer_ttl_steps と同水準)。"""
    sim = _sim(tmp_path, "tr_daily", n_steps=1, **ON)
    a = sim.agents[0]
    node = _seed(sim, a)
    T.phase(sim, 144, 1440)
    assert abs(T.strength_of(sim, node, "gathering") - 0.5) < 1e-12
    T.phase(sim, 288, 2880)
    assert abs(T.strength_of(sim, node, "gathering") - 0.25) < 1e-12


def test_transient_tier_decays_on_its_own_half_life(tmp_path):
    """transient = 半減期 18 step(~3 時間)= daily より 8 倍速く薄れる。"""
    sim = _sim(tmp_path, "tr_transient", n_steps=1, **ON,
               **{"world.traces.tier_of.gathering": "transient"})
    a = sim.agents[0]
    node = _seed(sim, a)
    T.phase(sim, 18, 1440)                         # 経過 18 step = 半減期ちょうど
    assert abs(T.strength_of(sim, node, "gathering") - 0.5) < 1e-12


def test_persistent_tier_never_decays(tmp_path):
    """persistent = 半減期 0 = 減衰しない(場所に定着した性格)。"""
    sim = _sim(tmp_path, "tr_persist", n_steps=1, **ON,
               **{"world.traces.tier_of.gathering": "persistent"})
    a = sim.agents[0]
    node = _seed(sim, a)
    for day in range(1, 8):
        T.phase(sim, 144 * day, 1440 * day)
    assert T.strength_of(sim, node, "gathering") == 1.0
    assert sim._trace_state["evaporated"] == 0


def test_evaporation_runs_once_per_day_only(tmp_path):
    """★日境界 1 回(第75 dunbar の前例)= 同じ日に何度呼んでも二重に減らない。"""
    sim = _sim(tmp_path, "tr_once", n_steps=1, **ON)
    a = sim.agents[0]
    node = _seed(sim, a)
    T.phase(sim, 144, 1440)
    once = T.strength_of(sim, node, "gathering")
    for step, sm in ((145, 1450), (200, 2000), (287, 2870)):
        T.phase(sim, step, sm)
        assert T.strength_of(sim, node, "gathering") == once, "同じ日に二重蒸発した"


def test_faded_traces_are_dropped_and_counted(tmp_path):
    """drop_threshold 未満はレコードごと削除(明示的失効ロジックの義務化)。"""
    sim = _sim(tmp_path, "tr_drop", n_steps=1, **ON,
               **{"world.traces.tier_of.gathering": "transient"})
    a = sim.agents[0]
    node = _seed(sim, a)
    T.phase(sim, 144, 1440)                        # 0.5^(144/18) = 0.0039 < 0.05
    assert T.strength_of(sim, node, "gathering") == 0.0
    assert sim._trace_state["nodes"] == {}, "薄れた痕跡のレコードが残っている"
    assert sim._trace_state["evaporated"] == 1
    assert T.line_for(sim, a) is None


def test_first_day_boundary_does_not_decay(tmp_path):
    """初回の日境界設定では減らさない(前回蒸発 step が無いので経過が測れない)。"""
    sim = _sim(tmp_path, "tr_first", n_steps=1, **ON)
    a = sim.agents[0]
    _isolate(sim, [a])
    _host(sim, a, step=5, sim_min=50)
    T.phase(sim, 5, 50)                            # ← ここが初回の日境界(day 0)
    assert T.strength_of(sim, a.node, "gathering") == 1.0


# =========================================================================== #
# (E) 最強 1 件のみ・no-fingerprint(検収基準 ④)
# =========================================================================== #
def test_only_the_strongest_single_trace_is_injected(tmp_path):
    sim = _sim(tmp_path, "tr_top", n_steps=1, **ON)
    a, later = sim.agents[0], sim.agents[1]
    _isolate(sim, [a])
    _host(sim, a, title="集まり1")
    _host(sim, a, title="集まり2")                  # gathering = 2.0
    _open(sim, a)                                  # opening   = 1.0(同じ node)
    T.phase(sim, 0, 0)
    node = a.node
    assert T.strength_of(sim, node, "gathering") == 2.0
    assert T.strength_of(sim, node, "opening") == 1.0
    _colocate(later, a)
    assert T.line_for(sim, later) == T.sentence("gathering")   # 最強 1 件だけ
    # 材料に入るのも 1 行(改行を含まない = 複数行に化けていない)
    line = scheduler._gather_material(sim, later, "solo", 1, 10)["trace_line"]
    assert line is not None and "\n" not in line


def test_ties_break_on_kind_name_ascending(tmp_path):
    """同点は種名昇順(決定論)= gathering < opening。"""
    sim = _sim(tmp_path, "tr_tie", n_steps=1, **ON)
    a, later = sim.agents[0], sim.agents[1]
    _isolate(sim, [a])
    _host(sim, a)
    _open(sim, a)
    T.phase(sim, 0, 0)
    assert T.strength_of(sim, a.node, "gathering") == \
        T.strength_of(sim, a.node, "opening") == 1.0
    _colocate(later, a)
    assert T.line_for(sim, later) == T.sentence("gathering")


_BANNED = ("silent", "memory", "engaged", "ablate", "placebo", "seed", "trace",
           "tier", "transient", "daily", "persistent", "strength", "node",
           "実験", "条件", "閾値", "発火", "驚き", "エピソード", "モデル", "強度",
           "シミュレーション", "エージェント", "アブレーション", "対照", "観測")


def test_template_and_words_have_no_digits_and_no_condition_words():
    """テンプレート**そのもの**と定型語に数字・実験条件語・機構語が 1 文字も無い。"""
    bare = T.TEXT.replace("{word}", "")
    assert not re.search(r"[0-9０-９]", bare)
    for w in _BANNED:
        assert w not in bare, f"テンプレに禁止語 {w} が入った"
    for kind, (_srcs, _tier, word) in T.TRACE_SPECS.items():
        assert not re.search(r"[0-9０-９]", word), f"{kind}: 定型語に数字が入った"
        for w in _BANNED:
            assert w not in word, f"{kind}: 定型語に禁止語 {w} が入った"


def test_sentence_is_a_pure_function_of_kind():
    """文面は**痕跡種だけ**の純関数(強度も階層も場所名も入らない)。"""
    for kind in T.TRACE_SPECS:
        s = T.sentence(kind)
        assert s and not re.search(r"[0-9０-９]", s), kind
        for w in _BANNED:
            assert w not in s, f"{kind}: 禁止語 {w} が入った"
        assert s == T.sentence(kind)               # 何度呼んでも同一
    assert T.sentence("存在しない種") == ""         # 語彙外は捏造しない


def test_line_does_not_depend_on_experiment_conditions(tmp_path):
    """同じ痕跡種からは**全実験条件で同一バイト列**(no-fingerprint の機械固定)。"""
    variants = (
        {}, {"k.writeback": "off"}, {"k.writeback": "sham"},
        {"ablate.cognitive_tier": "small"}, {"controls.mode": "compute_matched"},
        {"cognition.rejection_notify": "memory"},
        {"information.rumors.enabled": "true"},
        {"world.traces.tier_of.gathering": "transient"},
        {"world.traces.half_life_steps.daily": 999},
        {"world.traces.max_strength": 9.0},
    )
    seen: set[str] = set()
    for i, extra in enumerate(variants):
        sim = _sim(tmp_path, f"tr_pf{i}", n_steps=1, **ON, **extra)
        a, later = sim.agents[0], sim.agents[1]
        _isolate(sim, [a])
        _host(sim, a)
        T.phase(sim, 0, 0)
        _colocate(later, a)
        seen.add(str(T.line_for(sim, later)))
    assert len(seen) == 1, f"条件によって痕跡の文面が変わった {seen}"


def test_strength_and_tier_never_reach_a_prompt(tmp_path):
    """強度・階層名・件数は観測側だけの概念 — プロンプトに 1 度も現れない。"""
    sim = _sim(tmp_path, "tr_leak", n_steps=216, n_agents=18, **ON)
    seen: list[str] = []
    inner = sim.llm.generate

    def _gen(prompt, **kw):
        seen.append(prompt)
        return inner(prompt, **kw)
    sim.llm.generate = _gen
    sim.run()
    assert _kind(sim, "trace_mark"), "テストが空振り(痕跡が 1 件も刻まれていない)"
    blob = "\n".join(seen)
    for tier in T.TIERS:
        assert tier not in blob, f"階層名 {tier} がプロンプトへ漏れた"
    for word in ("trace", "痕跡", "強度"):
        assert word not in blob, f"機構語 {word} がプロンプトへ漏れた"


# =========================================================================== #
# (F) LLM 呼数・プロンプト差分(検収基準 ⑦)
# =========================================================================== #
def test_llm_call_count_is_identical_and_the_diff_is_one_line(tmp_path):
    """★呼数が完全一致し、プロンプトの差分が **trace_line の 1 欄に閉じる**。"""
    def _run(name, **ov):
        sim = _sim(tmp_path, name, n_steps=216, n_agents=18, **ov)
        sim.llm = _FixedLLM(_HOST_AND_TALK)
        sim.run()
        return sim

    off = _run("tr_k_off", **OFF)
    on = _run("tr_k_on", **ON)
    assert on.llm.calls == off.llm.calls > 0, \
        f"痕跡 ON で呼数が動いた: on={on.llm.calls} off={off.llm.calls}"
    assert _kind(on, "trace_mark"), "テストが空振り(痕跡が 1 件も刻まれていない)"
    # 世界そのものは動かない: L1 の差は trace_mark の増分だけ
    assert [e for e in _l1x(on) if e[2] != "trace_mark"] == _l1x(off), \
        "痕跡が世界状態を動かしている(観測入力を足すだけの層のはず)"
    # プロンプトの差分は痕跡の定型文だけ(新しい欄も、既存欄の書き換えも無い)
    allowed = {T.sentence(k) for k in T.TRACE_SPECS}
    diff: set[str] = set()
    for pa, pb in zip(off.llm.prompts, on.llm.prompts):
        if pa == pb:
            continue
        la, lb = pa.splitlines(), pb.splitlines()
        diff |= (set(lb) - set(la)) | (set(la) - set(lb))
    assert diff, "ON なのにプロンプトが 1 行も変わっていない(空振り)"
    assert diff <= allowed, f"痕跡の定型文以外の差分が出た: {sorted(diff - allowed)}"
    # 各プロンプトで痕跡の行は**最大 1 行**
    for prompt in on.llm.prompts:
        assert sum(1 for ln in prompt.splitlines() if ln in allowed) <= 1


def test_llm_call_count_is_k_invariant(tmp_path):
    """compute_matched 下で k=free と k=off の呼数が完全一致(R1)。"""
    def _run(name, writeback):
        sim = _sim(tmp_path, name, n_steps=144, n_agents=15, **ON,
                   **{"controls.mode": "compute_matched", "k.writeback": writeback})
        sim.llm = _FixedLLM(_HOST_AND_TALK)
        sim.run()
        return sim
    free = _run("tr_kfree", "free")
    off = _run("tr_koff", "off")
    assert free.llm.calls == off.llm.calls > 0


# =========================================================================== #
# (G) 契約の追随(検収基準 ⑧)
# =========================================================================== #
def test_trace_line_is_declared_in_the_perception_contract():
    assert "trace_line" in PC.PROMPT_KEYWORDS
    assert "trace_line" in {f for _kw, f, _s in PC._KW_FIELDS}
    assert "trace_line" not in PC.NON_PROMPT_FIELDS


def test_contract_round_trip_keeps_the_trace_line(tmp_path):
    sim = _sim(tmp_path, "tr_contract", n_steps=1, **ON)
    a, later = sim.agents[0], sim.agents[1]
    _isolate(sim, [a])
    _host(sim, a)
    T.phase(sim, 0, 0)
    _colocate(later, a)
    material = scheduler._gather_material(sim, later, "solo", 1, 10)
    percept = scheduler.build_perception(sim, later, material)
    assert percept.trace_line == T.sentence("gathering")
    # 無損失(契約の中心)。material は _gather_material の生の出力なので、_llm_speak が
    # 後から足す 5 欄は比較の対象外にする(tests/test_physics_zones.py:666 と同じ流儀)。
    # ★trace_line は **_gather_material が集める側**(場所ラベル行と同じ族)なので、
    #   後から足す 5 欄には**含まれない**= 契約列挙の集合は IF-B から変わらない。
    kw = percept.prompt_kwargs()
    assert {k: v for k, v in kw.items() if k in material} == material
    assert set(kw) - set(material) == {"interstitial_digest", "watch_section",
                                       "revision_line", "engaged_section",
                                       "reject_line"}
    assert T.sentence("gathering") in percept.text_blob()


def test_build_prompt_adds_exactly_one_line(tmp_path):
    """既存の 1 行欄と完全同型の seam: None は 1 行も足さず、ON は 1 行だけ足す。"""
    sim = _sim(tmp_path, "tr_prompt", n_steps=1, **ON)
    a = sim.agents[0]
    kwargs = scheduler._gather_material(sim, a, "solo", 0, 0)
    kwargs.pop("trace_line")
    base = deliberate.build_prompt(a, **kwargs, trace_line=None)
    line = T.sentence("gathering")
    withline = deliberate.build_prompt(a, **kwargs, trace_line=line)
    assert deliberate.build_prompt(a, **kwargs) == base
    added = [ln for ln in withline.splitlines() if ln not in base.splitlines()]
    assert added == [line]
    assert len(withline.splitlines()) == len(base.splitlines()) + 1
    # 「この場所」を指す文なので、場所を名指しした行より**後**に出る
    lines = withline.splitlines()
    assert lines.index(line) > max(i for i, ln in enumerate(lines)
                                   if ln.startswith("場所: "))


# =========================================================================== #
# (H) 決定論・resume・要約(検収基準 ⑤⑥)
# =========================================================================== #
def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _sim(tmp_path, "tr_det_a", n_steps=288, n_agents=20, **ON)
    a.run()
    b = _sim(tmp_path, "tr_det_b", n_steps=288, n_agents=20, **ON)
    b.run()
    assert _l1x(a) == _l1x(b), "痕跡 ON の決定論が崩れている"
    assert _kind(a, "trace_mark"), "テストが空振り"
    assert a._trace_state["nodes"] == b._trace_state["nodes"]
    assert a._trace_state["lines"] == b._trace_state["lines"]


def test_a_full_on_run_marks_and_injects(tmp_path):
    """ラン全体でも集約と 1 行注入が実際に立つ(空振り検収を防ぐ)。"""
    sim = _sim(tmp_path, "tr_run", n_steps=432, n_agents=25, **ON)
    sim.run()
    assert _kind(sim, "trace_mark"), "ON ランで痕跡が 1 件も刻まれていない"
    summary = json.loads((tmp_path / "tr_run" / "summary.json")
                         .read_text(encoding="utf-8"))
    prov = summary["traces"]
    assert prov["marks"] == len(_kind(sim, "trace_mark"))
    assert prov["lines_injected"] > 0, "1 行も後続者のプロンプトへ入っていない"
    assert sum(prov["by_kind"].values()) == prov["marks"]
    assert set(prov["by_kind"]) <= set(T.TRACE_SPECS)
    assert prov["kinds"] == list(T.DEFAULT_KINDS)
    assert prov["tier_of"]["opening"] == "persistent"
    assert prov["strength_max"] <= 3.0


def test_resume_matches_straight(tmp_path):
    """痕跡 ON で resume==straight(parquet バイト一致 + 痕跡ストア一致)。"""
    ov = {**ON, "run.start_tod": "00:00", "run.natural_start": "true"}
    split, total = 144, 288
    straight_dir = tmp_path / "tr_straight"
    straight = Simulation(_cfg("tr_straight", total, 20, **ov), out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "trace_mark"), "テスト前提が崩れた(痕跡ゼロ)"

    d = tmp_path / "tr_resumed"
    sim1 = Simulation(_cfg("tr_resumed", split, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("tr_resumed", total, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(traces resume)"
    ja = json.loads((straight_dir / "summary.json").read_text(encoding="utf-8"))
    jb = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert ja["traces"] == jb["traces"], \
        "痕跡ストア/タリーが resume で straight と食い違う(中央管理の漏れ)"
    assert straight._trace_state["nodes"] == sim2._trace_state["nodes"]
