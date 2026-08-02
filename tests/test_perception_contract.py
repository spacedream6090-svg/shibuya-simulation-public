"""第85バッチ: Perception / Intent 契約(physics-instructions.md **Part P1**)のテスト。

正典: docs/plans/source/physics-instructions.md Part P1 /
      docs/plans/cognition-physics-plan.md §4 第85行。

守るもの(検収基準の順)
  (1) 既定 OFF = 純粋既定と L1 バイト一致(ゴールデンは tests/test_scenario.py が別途担保)
  (2) **契約 ON/OFF で世界状態が一致**: 同 seed 288 step の mock ランで
      L1/L2/L3 バイト一致・乱数 draw 一致・LLM 呼数一致
      (= P1(3)「この時点では挙動は変わらない」の機械的な定義)
  (3) 無損失性: `Perception.prompt_kwargs()` が材料 dict と完全に等しい /
      `Intent.from_action(a).to_action() == a`
  (4) 静的検査: プロンプト構築層が world state を直接読まない(真偽台帳の漏洩検査と同じ型)。
      残置している直接参照は**宣言リストと完全一致**することを固定する(黙って増やせない)
  (5) Perception に真偽台帳由来の値が混入しない(既存 canary 関門への統合)
  (6) resume(契約 ON)一致・registry / timeconv の宣言
検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import registry as R
from society import timeconv as T
from society import truth_ledger as TL
from society.cognition import deliberate
from society.cognition import perception_contract as PC
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

SRC = Path(__file__).resolve().parents[1] / "src" / "society"


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int = 48, n_agents: int = 20, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 48, n_agents: int = 20, **ov):
    out = tmp_path / name
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=out)
    summary = sim.run()
    return sim, out, summary


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


class _CountingHub:
    """stream 派生と draw 消費を数えるプロキシ(test_channels と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _funcs_referencing(path: Path, name: str) -> set[str]:
    """そのファイルの中で識別子 `name` を参照している関数名の集合(AST)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == name:
                out.add(node.name)
                break
    return out


# --------------------------------------------------------------------------- #
# (A) 型の健全性 — 契約が build_prompt の署名と噛み合っている
# --------------------------------------------------------------------------- #
def test_default_is_off():
    cfg = load_config()
    assert bool(cfg.cognition.contract.enabled) is False
    assert PC.build_cfg(None) == {"enabled": False}
    assert PC.build_cfg({"enabled": "yes"})["enabled"] is True


def test_prompt_keywords_cover_the_build_prompt_signature():
    """★契約の要: `build_prompt` のキーワードは 1 つ残らず Perception を通る。

    build_prompt に引数を足したのに契約へ足し忘れると、その情報だけが契約の**外**から
    プロンプトへ入る(= P1 の「直接参照ゼロ」が穴あきになる)。ここで機械的に塞ぐ。
    """
    sig = inspect.signature(deliberate.build_prompt)
    kws = {p for p in sig.parameters if p != "agent"}
    assert set(PC.PROMPT_KEYWORDS) == kws, (
        "build_prompt の署名と契約の対応表がずれている。"
        f" 契約に無い: {sorted(kws - set(PC.PROMPT_KEYWORDS))} /"
        f" 署名に無い: {sorted(set(PC.PROMPT_KEYWORDS) - kws)}")
    assert len(PC.PROMPT_KEYWORDS) == len(set(PC.PROMPT_KEYWORDS))


def _full_material() -> dict:
    """全キーワードに**非既定**の値を入れた材料(round-trip の抜けを検出するため)。"""
    material = {}
    for i, kw in enumerate(PC.PROMPT_KEYWORDS):
        material[kw] = f"v{i}:{kw}"
    material.update({
        "nearby_pois": ["店A", "店B"], "nearby_names": ["甲", "乙"],
        "nearby_ids": [3, 1], "scene_lines": ["見え1", "見え2"],
        "familiar_places": ["馴染み"], "institutions": ["取り決め"],
        "memberships": ["会"], "feed_texts": ["TL1", "TL2"],
        "dialog_history": [("甲", "やあ"), ("乙", "どうも")],
        "reply_to": ("甲", "こんにちは"),
        "step": 7, "sim_min": 430,
        "equip_all": True, "open_actions": True, "explicit_nothing": True,
        "verify_actions": True, "variety_hint": True, "venture_cost": 12345.0,
        "labeling_mode": "open",
    })
    return material


def test_perception_round_trip_is_lossless():
    """★P1 の等価性の定義: 材料 → Perception → 材料 が**完全に**同じ dict になる。"""
    material = _full_material()
    percept = PC.Perception.from_material(material)
    got = percept.prompt_kwargs()
    assert got == material, "Perception 経由で材料が変わった(プロンプトが変わる)"
    assert set(got) == set(material)
    for key, value in material.items():          # 型まで一致(list が tuple 化しない)
        assert type(got[key]) is type(value), f"{key}: 型が変わった"


def test_round_trip_survives_all_none_material():
    """既定値だらけ(None / 空)の材料でも None と [] を取り違えない。"""
    material = {kw: None for kw in PC.PROMPT_KEYWORDS}
    material.update({"place_name": "", "nearby_pois": [], "nearby_names": [],
                     "nearby_ids": [], "step": 0, "sim_min": 0,
                     "labeling_mode": "constrained", "city_name": "",
                     "open_actions": False, "explicit_nothing": False,
                     "verify_actions": False, "equip_all": False,
                     "variety_hint": False, "venture_cost": 30000.0})
    percept = PC.Perception.from_material(material)
    got = percept.prompt_kwargs()
    assert got == material
    assert got["scene_lines"] is None and got["nearby_pois"] == []


def test_unknown_material_is_refused():
    """契約に無いキーは黙って捨てず即エラー(情報が消える穴を作らない)。"""
    with pytest.raises(KeyError):
        PC.Perception.from_material({"place_name": "渋谷", "mystery_line": "x"})


def test_non_prompt_fields_never_reach_the_prompt():
    """身体・内部・salience は**プロンプトに 1 つも出ない**(P3 の no-fingerprint 前倒し)。"""
    percept = PC.Perception.from_material(
        _full_material(),
        body={"local_density": 9.9, "blocked": True},
        internal={"drive": 0.5},
        salience={"crowd_line": 3.0})
    kwargs = percept.prompt_kwargs()
    for name in PC.NON_PROMPT_FIELDS:
        assert name not in kwargs, f"{name} がプロンプト引数へ漏れている"
    assert "9.9" not in json.dumps(kwargs, ensure_ascii=False, default=str)
    # それでも観測側からは読める(P0 の顕著性発火・P3 の物理写像の入口)
    assert percept.salience_of("crowd_line") == 3.0
    assert percept.salience_of("weather_line") is None      # 欠測は欠測
    assert percept.body["blocked"] is True


def test_perception_is_the_only_type_and_is_immutable():
    percept = PC.Perception.from_material(_full_material())
    with pytest.raises(Exception):
        percept.place_name = "書き換え"      # frozen dataclass


def test_text_blob_collects_every_string_material():
    material = _full_material()
    blob = PC.Perception.from_material(material).text_blob()
    for value in ("店A", "甲", "やあ", "こんにちは", "見え1", "取り決め"):
        assert value in blob, f"漏洩検査の対象から {value} が落ちている"
    assert material["ads_line"] in blob and material["weather_line"] in blob


# --------------------------------------------------------------------------- #
# (B) salience — 第80 channels の値を再利用(二重計算しない)
# --------------------------------------------------------------------------- #
def test_salience_map_points_at_real_channels():
    from society.cognition import channels as CH
    known = {c.id for c in CH.CHANNELS}
    unknown = sorted(set(PC.SALIENCE_MAP) - known)
    assert unknown == [], f"存在しないチャンネル id を写している: {unknown}"
    targets = {f.name for f in __import__("dataclasses").fields(PC.Perception)}
    for cid, fields_ in PC.SALIENCE_MAP.items():
        for fname in fields_:
            assert fname in targets, f"{cid} → 未知の項目 {fname}"
    # factors 層のキーを含む body.state.* は写さない(no-fingerprint)
    assert not [c for c in PC.SALIENCE_MAP if c.startswith("body.state.")]


def test_salience_takes_the_max_and_drops_unmapped():
    got = PC.salience_from_channels({"ext.crowd_local": 1.0, "ext.encounter": 2.5,
                                     "pred.unmet": 9.0, "body.state.x": 4.0})
    assert got["nearby_names"] == 2.5, "複数チャンネルが写る項目は最大を採る"
    assert got["crowd_line"] == 1.0 and got["nearby_ids"] == 2.5
    assert "internal" not in got                    # 写像先の無い入力は捨てる
    assert PC.salience_from_channels({}) == {}      # 観測が無ければ空 = 欠測


def test_salience_is_empty_without_observations(tmp_path):
    """channels/fire が OFF のランでは salience を捏造しない(0.0 で埋めない)。"""
    sim = Simulation(_cfg("sal_off", n_steps=1, n_agents=6), out_dir=tmp_path / "sal")
    assert scheduler._percept_salience(sim, sim.agents[0]) == {}


# --------------------------------------------------------------------------- #
# (C) Intent — 思考の出力の唯一の型
# --------------------------------------------------------------------------- #
_ACTION_SAMPLES = (
    '{"action":"speak","text":"こんにちは","use_terms":["語"]}',
    '{"action":"coin_label","word":"新語","text":"新語を使う"}',
    '{"action":"post","text":"つぶやき"}',
    '{"action":"dm","text":"メッセージ"}',
    '{"action":"host_event","title":"勉強会","hours_later":3}',
    '{"action":"post_flyer","text":"貼り紙"}',
    '{"action":"found_group","name":"会","purpose":"目的"}',
    '{"action":"propose","text":"提案","rule":{"kind":"x"}}',
    '{"action":"job_search"}',
    '{"action":"open_venture","name":"屋台","offer":"たこ焼き","permit":false}',
    '{"action":"move_home","area":"代々木"}',
    '{"action":"buy","cat":"food"}',
    '{"action":"study","topic":"英語"}',
    '{"action":"propose_partnership","to":"乙"}',
    '{"action":"break_up"}',
    '{"action":"do","what":"散歩する","where":"公園","minutes":45,"value":["感情"]}',
    '{"action":"verify","about":"噂","how":"go"}',
    '{"action":"plan","items":[{"when":"朝","what":"work","place":"店"}]}',
    '{"action":"wander"}',
    '{"action":"nothing"}',
    '{"action":"recall","query":"昨日"}',
    '{"action":"reflect","summary":"要約","belief":"考え",'
    '"salient":[{"text":"出来事","importance":7}],"self":"自分","ties":"関係"}',
)


@pytest.mark.parametrize("response", _ACTION_SAMPLES)
def test_intent_round_trip_is_lossless(response):
    """★P1 の等価性: 行動 dict → Intent → 行動 dict が**完全復元**(情報等価)。"""
    action = deliberate.parse_action(response)
    assert action is not None, f"サンプルが解釈できない: {response}"
    back = PC.Intent.from_action(action).to_action()
    assert back == action, "Intent 経由で行動の情報が落ちた"
    assert list(back) == list(action), "キーの並びまで含めて同一でなければならない"


def test_intent_covers_every_known_action_verb():
    """サンプルが `KNOWN_ACTIONS` を(別名を除いて)覆っていることの自己点検。"""
    kinds = {deliberate.parse_action(r)["type"] for r in _ACTION_SAMPLES}
    assert {"speak", "post", "dm", "stay", "plan", "reflect", "verify",
            "free_action", "coin_label"} <= kinds


def test_intent_of_none_is_none():
    assert PC.Intent.from_action(None) is None


def test_intent_has_no_execution_details():
    """★P1(2): Intent に経路の各点・速度の時系列を含めない。"""
    import dataclasses
    names = " ".join(f.name for f in dataclasses.fields(PC.Intent))
    bad = [t for t in PC.FORBIDDEN_INTENT_TERMS
           if re.search(rf"(^|_){re.escape(t)}(_|$)", names)]
    assert bad == [], f"Intent が物理的な実行の詳細を持っている: {bad}"


def test_intent_projects_the_contract_fields():
    """移動目標・対話意図・話題・滞在意図が型の欄に立つ(dict を舐めなくても読める)。"""
    walk = PC.Intent.from_action(deliberate.parse_action(
        '{"action":"do","what":"散歩する","where":"代々木公園","minutes":45}'))
    assert walk.move_poi == "代々木公園" and walk.stay is False
    talk = PC.Intent.from_action(deliberate.parse_action(
        '{"action":"propose_partnership","to":"乙"}'))
    assert talk.talk_to == "乙"
    verify = PC.Intent.from_action(deliberate.parse_action(
        '{"action":"verify","about":"噂の店","how":"go"}'))
    assert verify.topic == "噂の店"
    stay = PC.Intent.from_action(deliberate.parse_action('{"action":"wander"}'))
    assert stay.stay is True and stay.kind == "stay"
    # 現行の行動スキーマに欄が無い量は既定値のまま(捏造しない)
    assert walk.urgency == 0.0 and walk.avoidance == 0.0 and walk.move_xy is None


def test_intent_refuses_a_non_action():
    with pytest.raises(TypeError):
        PC.Intent.from_action({"no_type": 1})


# --------------------------------------------------------------------------- #
# (D) R1 — 既定 OFF バイト一致 / 契約 ON/OFF で世界一致(★検収基準 1・2)
# --------------------------------------------------------------------------- #
def test_off_is_byte_identical_to_pure_default(tmp_path):
    a, _o, _s = _run(tmp_path, "d_pure", 144, 15)
    b, _o, _s = _run(tmp_path, "d_off", 144, 15,
                     **{"cognition.contract.enabled": "false"})
    assert _l1(a) == _l1(b), "既定 OFF が純粋既定と一致しない"


def test_contract_on_off_produce_the_same_world(tmp_path):
    """★P1 の受け入れ基準: 契約経路 ON/OFF で世界状態が一致(288 step mock)。"""
    off = Simulation(_cfg("c_off", 288, 30), out_dir=tmp_path / "c_off")
    off.hub = _CountingHub(off.hub)
    off.run()
    on = Simulation(_cfg("c_on", 288, 30,
                         **{"cognition.contract.enabled": "true"}),
                    out_dir=tmp_path / "c_on")
    on.hub = _CountingHub(on.hub)
    on.run()

    assert _l1(off) == _l1(on), "契約 ON で L1 が変わった(プロンプトが変わっている)"
    assert [dict(m) for m in off.logger.metrics] == \
           [dict(m) for m in on.logger.metrics], "契約 ON で L2 が変わった"
    assert [dict(s) for s in off.logger.snapshots] == \
           [dict(s) for s in on.logger.snapshots], "契約 ON で L3 が変わった"
    assert off.hub.counts == on.hub.counts, \
        f"乱数 draw が変わった: {off.hub.counts} vs {on.hub.counts}"
    assert off.llm.calls == on.llm.calls > 0, \
        f"LLM 呼数が変わった: {off.llm.calls} vs {on.llm.calls}"
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        pa = (tmp_path / "c_off" / f"{stem}.parquet")
        pb = (tmp_path / "c_on" / f"{stem}.parquet")
        assert pq.read_table(pa).to_pylist() == pq.read_table(pb).to_pylist(), \
            f"{stem} が契約 ON/OFF で不一致"


def test_prompts_are_byte_identical_on_and_off(tmp_path):
    """等価性の**定義**そのもの: 全プロンプト文字列が 1 バイトも変わらない。"""
    from society.llm.journal import iter_records
    ov = {"model.journal": "true"}
    _a, out_a, _s = _run(tmp_path, "p_off", 144, 20, **ov)
    _b, out_b, _s = _run(tmp_path, "p_on", 144, 20,
                         **{**ov, "cognition.contract.enabled": "true"})
    pa = [r["prompt"] for r in iter_records(out_a / "llm_journal.jsonl.gz")]
    pb = [r["prompt"] for r in iter_records(out_b / "llm_journal.jsonl.gz")]
    assert pa and pa == pb, "契約経路でプロンプト文字列が変わった"


def test_contract_on_is_deterministic(tmp_path):
    a, _o, _s = _run(tmp_path, "det_a", 144, 20,
                     **{"cognition.contract.enabled": "true"})
    b, _o, _s = _run(tmp_path, "det_b", 144, 20,
                     **{"cognition.contract.enabled": "true"})
    assert _l1(a) == _l1(b), "契約 ON が非決定(乱数を引いている疑い)"


def test_contract_on_with_channels_and_fire_still_matches(tmp_path):
    """salience の源(第80/81)が ON のランでも契約 ON/OFF は世界一致。

    ここが緑なら「salience を読むことが世界に触れていない」ことが実測で固定される。
    """
    ov = {"cognition.channels.enabled": "true", "cognition.fire.enabled": "true"}
    off, _o, _s = _run(tmp_path, "f_off", 144, 24, **ov)
    on, _o, _s = _run(tmp_path, "f_on", 144, 24,
                      **{**ov, "cognition.contract.enabled": "true"})
    assert _l1(off) == _l1(on)
    assert off.llm.calls == on.llm.calls > 0


def test_perception_carries_salience_when_channels_are_on(tmp_path):
    """空回り防止: fire ON のランで salience が実際に値を持つ個体が出る。"""
    sim = Simulation(_cfg("sal_on", 144, 24,
                          **{"cognition.channels.enabled": "true",
                             "cognition.fire.enabled": "true",
                             "cognition.contract.enabled": "true"}),
                     out_dir=tmp_path / "sal_on")
    sim.run()
    seen = [scheduler._percept_salience(sim, a) for a in sim.agents]
    assert any(s for s in seen), \
        "fire ON でも salience が 1 件も立たない(channels 再利用の配線が死んでいる)"


# --------------------------------------------------------------------------- #
# (E) 静的検査 — プロンプト構築層は world state を直接読まない(★検収基準 4)
# --------------------------------------------------------------------------- #
# 「プロンプト構築層」= プロンプト文字列を組み立てる module。
_PROMPT_LAYER = ("deliberate.py", "perception_contract.py", "reflection.py")

# (a) プロンプト構築層に残る直接参照。**関数名まで**宣言する(P3 前に潰す対象)。
_DECLARED_WORLD_READERS: dict[str, tuple[str, ...]] = {
    # プロンプト構築層の中に残る 2 関数。どちらもプロンプト文字列を組まない
    # 「方針キャッシュへの薄い seam」で、sim は素通しの引数(内容は policy_cache が読む)。
    "deliberate.py": ("maybe_reuse_action", "store_action"),
    "perception_contract.py": (),
    "reflection.py": (),
}

# (b) 朝の一日計画。`build_plan_prompt` 自体は純関数(材料は引数)だが、材料集めと発行が
#     同居しているため世界を直接読む。**P3 前に契約経路へ寄せる第一候補**(発話経路と
#     同じ形 = _gather_material → Perception → build_plan_prompt にできる)。
_PLANNING_WORLD_READERS = ("_make_plan_framework", "_today_schedule_line",
                           "apply_plan_response", "build_plan_request", "make_plan")

# (c) プロンプトを組まない層(観測・発火・行動選択・較正・方針キャッシュ)。世界を読むのが
#     本務なので**関数名までは凍結しない**(無関係のリファクタで fail させない)。
#     固定するのは「どの module が世界を読むか」の集合だけ = 世界を読む新しい module が
#     認知層に増えたら気づける。
_NON_PROMPT_WORLD_READER_MODULES = frozenset({
    "calib.py", "channels.py", "fire.py", "plan_schema.py", "plasticity.py",
    "policy_cache.py", "routine.py", "watch.py",
    # 第86バッチ day_plan v1: 計画の**物理検証**(移動時間・営業時間・場所カテゴリの実在)と
    # **場所の具体解決**(習慣・距離・営業状態)は、定義上 world(city の POI・座標・
    # commerce の営業時刻)を読まないと成立しない。プロンプト構築層ではない(schema_prompt は
    # cfg しか見ない)ので、routine.py と同じ扱いで台帳へ載せる。P3 で契約経路へ寄せる対象。
    "day_plan.py",
})


def test_build_prompt_never_reads_world_state():
    """★P1 受け入れ基準: プロンプト構築コードから world state への直接参照ゼロ。"""
    path = SRC / "cognition" / "deliberate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    target = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "build_prompt")
    names = {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    forbidden = names & {"sim", "world", "city", "logger"}
    assert forbidden == set(), \
        f"build_prompt が世界を直接読んでいる: {sorted(forbidden)}"
    # 参照するのは引数と agent(自分自身)だけ = 契約の外から情報が入る口が無い
    args = {a.arg for a in target.args.args} | {a.arg for a in target.args.kwonlyargs}
    assert "agent" in args and set(PC.PROMPT_KEYWORDS) <= args


def test_contract_module_never_touches_the_world():
    """契約の型 module は `sim` を 1 度も参照せず、world / engine を import しない。"""
    text = (SRC / "cognition" / "perception_contract.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    # 識別子としての `sim`(変数・引数)が 1 つも無い(散文での言及は対象外)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    assert "sim" not in used, \
        "契約 module が world オブジェクトを参照している(生成は世界側の責務)"
    mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
             for a in n.names]
    assert all(not m.startswith(("..world", "..engine", "society")) and
               "world" not in m and "engine" not in m for m in mods), \
        f"契約 module が世界層を import している: {mods}"


def test_prompt_layer_has_no_undeclared_world_reads():
    """プロンプト構築層の直接参照は宣言済みのものだけ(deliberate の 2 seam のみ)。"""
    for name in _PROMPT_LAYER:
        got = _funcs_referencing(SRC / "cognition" / name, "sim")
        want = set(_DECLARED_WORLD_READERS.get(name, ()))
        assert got == want, \
            f"{name}: プロンプト構築層の world 直接参照が宣言と違う {sorted(got)}"


def test_planning_world_reads_are_declared():
    """朝の一日計画に残る直接参照(P3 前に契約経路へ寄せる第一候補)を関数名まで固定。"""
    got = _funcs_referencing(SRC / "cognition" / "planning.py", "sim")
    assert got == set(_PLANNING_WORLD_READERS), \
        f"planning.py の world 直接参照が宣言と違う: {sorted(got)}"


def test_residual_world_reading_modules_do_not_grow():
    """★残置リストの固定: 認知層で世界を直接読む module の集合が宣言と完全一致。

    P1 は「直接参照が残っていないことを静的検査で確認する」を要求するが、認知層には
    観測・発火・行動選択の module も同居している。それらを『プロンプト構築層ではない』
    と口頭で言い訳するのではなく、**台帳に書いて固定**する(世界を読む module が黙って
    増えない = P3 までに潰す対象が把握できる)。関数名まで凍結するのはプロンプト構築層と
    planning.py だけで、その他は module 粒度に留める(無関係な変更で fail させない)。
    """
    got = {path.name for path in sorted((SRC / "cognition").glob("*.py"))
           if _funcs_referencing(path, "sim")}
    want = (set(_NON_PROMPT_WORLD_READER_MODULES) | {"planning.py"}
            | {k for k, v in _DECLARED_WORLD_READERS.items() if v})
    assert got == want, (
        "cognition/ で世界を直接読む module が宣言リストとずれている。"
        " 契約経由へ移すか、理由つきで台帳へ足すこと。"
        f" 未宣言={sorted(got - want)} / 消えた={sorted(want - got)}")


def test_deliberate_imports_stay_minimal():
    """プロンプト構築 module の import は「気分の文言」だけ(世界層を引かない)。"""
    tree = ast.parse((SRC / "cognition" / "deliberate.py").read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    mods = {(n.module or "") for n in top if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {"json", "__future__", "factors.mood"}, \
        f"deliberate が余計な層を import している: {sorted(mods)}"


def test_perception_generation_lives_on_the_world_side():
    """★P1(1): Perception の生成関数は世界側(engine)に 1 本だけ。"""
    assert callable(scheduler.build_perception)
    src = inspect.getsource(scheduler.build_perception)
    assert "Perception.from_material" in src
    # cognition/ 側に `from_material` を呼ぶ場所は 1 つも無い(生成の唯一性)
    callers = [p.name for p in (SRC / "cognition").glob("*.py")
               if "from_material(" in p.read_text(encoding="utf-8")
               and p.name != "perception_contract.py"]
    assert callers == [], f"認知層が Perception を自作している: {callers}"


# --------------------------------------------------------------------------- #
# (F) 漏洩検査への統合(★検収基準 5)
# --------------------------------------------------------------------------- #
def test_check_percept_detects_a_ledger_canary():
    """Perception も既存 canary 関門を通る(台帳 → 世界情報 → プロンプトの経路を塞ぐ)。"""
    TL.reset_canaries()
    try:
        clean = PC.Perception.from_material(_full_material())
        TL.check_percept(clean)                       # 未武装=素通り
        token = TL.register_canary("f09999")
        TL.check_percept(clean)                       # 混入なし=素通り
        leaked = PC.Perception.from_material(
            {**_full_material(), "crowd_line": f"人が多い {token}"})
        with pytest.raises(TL.LedgerLeak) as ei:
            TL.check_percept(leaked)
        assert "f09999" in str(ei.value)
        # 列の中に沈んでいても見つける(scene_lines は list[str])
        deep = PC.Perception.from_material(
            {**_full_material(), "scene_lines": ["ふつうの見え", f"{token}"]})
        with pytest.raises(TL.LedgerLeak):
            TL.check_percept(deep)
    finally:
        TL.reset_canaries()


def test_no_canary_in_any_prompt_of_a_contract_run(tmp_path):
    """契約 ON + beliefs ON の実ランで、全プロンプトに台帳 canary が現れない。"""
    from society.llm.journal import iter_records
    sim, out, summary = _run(tmp_path, "c_canary", 144, 40,
                             **{"beliefs.enabled": "true",
                                "beliefs.verify_actions": "true",
                                "model.journal": "true",
                                "cognition.contract.enabled": "true"})
    facts = TL.facts_of(sim)
    assert facts, "fact が 1 件も積まれていない(テストの前提が崩れた)"
    canaries = [f["canary"] for f in facts.values()]
    recs = list(iter_records(out / "llm_journal.jsonl.gz"))
    assert len(recs) == summary["llm_calls"] > 0
    for rec in recs:
        assert TL.CANARY_PREFIX not in rec["prompt"]
        for tok in canaries:
            assert tok not in rec["prompt"]


# --------------------------------------------------------------------------- #
# (G) resume(★検収基準 4)
# --------------------------------------------------------------------------- #
def test_resume_matches_straight_run_with_contract_on(tmp_path):
    ov = {"cognition.contract.enabled": "true"}
    straight_dir = tmp_path / "r_straight"
    straight = Simulation(_cfg("r_straight", 120, 20, **ov), out_dir=straight_dir)
    straight.run()
    d = tmp_path / "r_resumed"
    split, total = 72, 120
    sim1 = Simulation(_cfg("r_resumed", split, 20,
                           **{**ov, "observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("r_resumed", total, 20,
                           **{**ov, "observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        a = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        b = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert a == b, f"{stem} 不一致(契約 ON の resume)"


# --------------------------------------------------------------------------- #
# (H) 宣言(registry / timeconv)
# --------------------------------------------------------------------------- #
def test_feature_is_declared_in_the_registry():
    feature = {f.id: f for f in R.FEATURES}.get("cognition.contract.enabled")
    assert feature is not None, "機能レジストリに宣言が無い"
    assert feature.repro_tier == "strict", "通り道を変えるだけ=乱数も LLM 呼も足さない"
    assert feature.affects_k is False, "LLM 呼び出し点を 1 つも増減しない"
    assert feature.fingerprint_risk == "none", "プロンプトを 1 バイトも変えない"


def test_verify_mode_keeps_the_contract_available():
    """strict 等級なので run.mode=verify(対照実験)でも自動 OFF にされない。"""
    cfg = load_config(["run.mode=verify", "cognition.contract.enabled=true"])
    report = R.describe(cfg)
    assert bool(cfg.cognition.contract.enabled) is True
    assert "cognition.contract.enabled" in {f["id"] for f in report["enabled"]}


def test_no_new_dt_dependent_conf_keys():
    """本バッチの新キーは bool 1 個のみ = Δt 分類の対象になる量を足していない。"""
    assert T.classify("cognition.contract.enabled") is None
    cfg = load_config()
    assert set(cfg.cognition.contract.keys()) == {"enabled"}
