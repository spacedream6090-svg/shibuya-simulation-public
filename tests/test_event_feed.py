"""F1 イベントフィード + F3 在館者表示改善(第137・レーンF)の検証。

これは **観測側だけ**の機能である。シミュレーションは 1 バイトも変わらず、変わるのは
「生成される HTML に何が描かれるか」だけ(src/ には一切触れていない)。検収条件:

1. 単一レジストリ(F1-1): `viz/notable_events.KIND_REGISTRY` から畳んだ
   NOTABLE_KINDS / MAGNITUDE_KEYS / HIGHLIGHT_KINDS が **統合前の直書きと 1 文字も違わない**。
   3D 顕著パネル(NOTABLE_KINDS)もライブ・ティッカー(HIGHLIGHT_KINDS)も、統合で挙動が
   変わってはいけない。`scripts/live_viewer.py` は集合を直書きせずレジストリを import する。
2. ランキング(F1-2): memo §4-1 の式が効いていること。とくに
   **希少 kind が高頻度の配管 kind より上位**(自己較正の希少度が働く)。
   決定論(同じ入力 → 同じ出力)・dedup・初出ボーナス・ペーシングの単調性。
3. 後方互換(最重要): `--event-feed` / `--indoor-v2` を **明示しなければ**生成 HTML は
   従来とバイト同一。フィード対象イベントが 0 件のランは明示しても注入されない。
4. F3 幾何: 散布点が **実フットプリント多角形の内側**に落ち、**時刻に依存しない**
   (sin 微動の撤去)。旧実装が同じ L 字建物で外に出ることも対照で示す(偽合格の防止)。

全経路 合成データのみ(実 LLM 不使用・乱数不使用)。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))

import make_viewer as mv  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NB = _load("nb_registry", "viz/notable_events.py")
FR = _load("feed_rank_t", "viz/feed_rank.py")


# ===========================================================================
# 1. 単一レジストリ — 統合前の直書きと同内容(3D/ライブの挙動を変えていない)
# ===========================================================================
# 統合前(第136 HEAD)の viz/notable_events.NOTABLE_KINDS そのもの。
FROZEN_NOTABLE = {
    "disaster":         ("災害", 5, ["kind", "phase"]),
    "world_event":      ("世界イベント", 5, ["title", "word"]),
    "scenario_shock":   ("摂動シナリオ", 5, ["kind", "phase"]),
    "annual_event":     ("年中行事", 4, ["name", "date"]),
    "infra_outage":     ("インフラ障害", 4, ["kind", "phase"]),
    "transit_delay":    ("交通の遅延・運休", 4, ["line", "kind"]),
    "crowd_surge":      ("大規模群集", 4, ["level", "event"]),
    "council_elected":  ("議会改選", 5, ["term", "members"]),
    "election_result":  ("選挙開票", 5, ["elected", "votes"]),
    "candidacy":        ("立候補", 4, ["day", "deposit"]),
    "ordinance_vote":   ("条例議決", 4, ["passed", "yes", "no"]),
    "proposal":         ("提案・署名運動", 3, ["text"]),
    "proposal_passed":  ("提案の成立", 4, ["text", "supporters"]),
    "institution":      ("制度化", 4, ["name", "norm_text"]),
    "institution_rule": ("ルールの制定", 4, ["name", "type"]),
    "rule_repealed":    ("ルールの廃止", 4, ["name", "type"]),
    "group_found":      ("コミュニティ結成", 4, ["name", "purpose"]),
    "event_host":       ("イベント開催", 3, ["title", "place"]),
    "flyer_post":       ("掲示・ビラ", 3, ["text"]),
    "venture_open":     ("出店・開業", 4, ["name", "offer"]),
    "venture_fulltime": ("起業(本業化)", 4, ["name"]),
    "labor_action":     ("労働争議", 4, ["org", "demand"]),
    "vc_investment":    ("ベンチャー出資", 4, ["venture", "amount"]),
    "bankruptcy":       ("自己破産", 5, ["debt"]),
    "eviction":         ("立退き/再入居", 4, ["phase", "arrears"]),
    "detention":        ("勾留", 4, ["target", "steps"]),
    "enforcement":      ("制度の執行", 4, ["target", "penalty"]),
    "crime":            ("犯罪・被害", 4, ["kind", "victim", "offender"]),
    "partner_formed":   ("交際成立", 4, ["other"]),
    "life_event":       ("ライフイベント", 4, ["kind", "other"]),
    "job_change":       ("転職・異動", 4, ["from_org", "to_org"]),
    "unemployment":     ("失業・求職", 4, ["state", "org"]),
    "long_goal":        ("人生目標の設定", 4, ["goal"]),
    "illness":          ("病気", 3, ["state", "kind"]),
    "medical_visit":    ("医療機関の受診", 3, ["cost"]),
    "label_coin":       ("新語の創出", 4, ["text"]),
    "label_adopt":      ("新語の普及", 2, ["item_id"]),
    "viral_cascade":    ("バイラル拡散", 3, ["reach"]),
    "misinfo":          ("誤情報・炎上", 3, ["kind"]),
    "nuisance":         ("迷惑行為", 2, ["kind"]),
}
FROZEN_MAG = {"viral_cascade": "reach", "proposal_passed": "supporters",
              "vc_investment": "amount", "bankruptcy": "debt", "crime": "amount"}
# 統合前の scripts/live_viewer.HIGHLIGHT_KINDS そのもの。
FROZEN_HIGHLIGHT = {
    "vocab_coin", "label_coin", "label_adopt", "place_label_bind",
    "joint_invite", "joint_activity", "undefined_action", "free_action",
    "belief_update", "belief_transmit", "belief_verify",
    "institution", "institution_rule", "proposal", "proposal_passed",
    "group_found", "group_join", "venture_open", "venture_close", "event_host",
    "flyer_post", "world_event", "scenario_shock",
    "relation_tier", "relation_break", "relation_dormant", "relation_rekindle",
    "labor_action", "candidacy", "election_result", "ordinance_vote",
    "move_home", "long_goal", "chance_event", "fallback",
}


def test_notable_kinds_unchanged_by_unification():
    """3D 顕著パネルの対応表は統合前と完全一致(順序も含む)。"""
    assert NB.NOTABLE_KINDS == FROZEN_NOTABLE
    assert list(NB.NOTABLE_KINDS) == list(FROZEN_NOTABLE)
    assert NB.MAGNITUDE_KEYS == FROZEN_MAG


def test_highlight_kinds_unchanged_by_unification():
    """ライブ・ティッカーの見せ場集合は統合前と完全一致。"""
    assert NB.HIGHLIGHT_KINDS == FROZEN_HIGHLIGHT


def test_live_viewer_imports_registry_not_a_literal():
    """live_viewer は集合を直書きせず、レジストリ由来の値を持つ(二重管理の解消)。"""
    lv = _load("live_viewer_t", "scripts/live_viewer.py")
    assert lv.HIGHLIGHT_KINDS == NB.HIGHLIGHT_KINDS == FROZEN_HIGHLIGHT
    assert tuple(lv._TEXT_KEYS) == tuple(NB.GENERIC_TEXT_KEYS)
    src = (REPO_ROOT / "scripts" / "live_viewer.py").read_text(encoding="utf-8")
    # 「HIGHLIGHT_KINDS = {」で始まる集合リテラルが残っていない(= 復活したら落ちる)
    assert not re.search(r"HIGHLIGHT_KINDS\s*=\s*\{\s*\n\s*\"", src), \
        "live_viewer に kind 集合リテラルが復活している(単一レジストリの回帰)"


def test_feed_kinds_is_union_of_both_views():
    """フィードの母集合 = 2 系統の和集合(どちらの視点も落とさない)。"""
    assert NB.FEED_KINDS == set(NB.NOTABLE_KINDS) | NB.HIGHLIGHT_KINDS
    assert len(NB.FEED_KINDS) == 58
    # レジストリの全 kind に label / importance / icon が揃っている
    for k, v in NB.KIND_REGISTRY.items():
        assert v["label"] and 1 <= v["importance"] <= 5 and v["icon"], k


def test_storyline_key_pairs_and_items():
    assert NB.storyline_key("label_adopt", 3, {"item_id": "v-1"}) == "item_id:v-1"
    # ペアは順序非依存(A→B と B→A が同じ物語になる)
    a = NB.storyline_key("partner_formed", 7, {"other": 2})
    b = NB.storyline_key("partner_formed", 2, {"other": 7})
    assert a == b == "pair:2-7"
    assert NB.storyline_key("illness", 1, {"state": "onset"}) is None   # story 指定なし
    assert NB.storyline_key("move_segment", 1, {}) is None              # 未登録 kind


# ===========================================================================
# 2. ランキング(feed_rank)
# ===========================================================================
def _ev(step, aid, kind, payload=None, x=10.0, y=5.0):
    return {"step": step, "sim_min": 420 + step * 10, "agent_id": aid, "kind": kind,
            "x": float(x), "y": float(y),
            "payload": json.dumps(payload or {}, ensure_ascii=False)}


def test_rarity_beats_frequency_at_equal_importance():
    """**同じ重要度**なら、希少な kind が高頻度の kind より必ず上位に来る。

    これが「配管:物語 = 1000:1 の選別」(memo §2-3)の核。importance を揃えることで
    効いているのが希少度であることを分離して示す。
    """
    rows = [_ev(s, s % 5, "flyer_post", {"text": f"告知{s}"}) for s in range(300)]
    rows += [_ev(11, 1, "misinfo", {"kind": "炎上", "item_id": "m1"}),
             _ev(203, 2, "misinfo", {"kind": "炎上", "item_id": "m2"})]
    assert NB.KIND_REGISTRY["flyer_post"]["importance"] == \
        NB.KIND_REGISTRY["misinfo"]["importance"] == 3
    recs = FR.score_events(FR.collect(rows))
    best = {}
    for r in recs:
        best[r["kind"]] = max(best.get(r["kind"], -9e9), r["score"])
    assert best["misinfo"] > best["flyer_post"], \
        "希少 kind が高頻度 kind を上回っていない(自己較正の希少度が効いていない)"
    # 順位でも: 上位 2 件が misinfo
    top2 = sorted(recs, key=lambda r: -r["score"])[:2]
    assert {r["kind"] for r in top2} == {"misinfo"}


def test_rare_story_kind_outranks_high_volume_plumbing_kind():
    """物語級の希少 kind が、配管的に大量発生する kind を押しのけて上位に来る。"""
    rows = [_ev(s, s % 7, "belief_transmit",
                {"claim": "うわさ", "item_id": f"b{s % 3}", "hop": 1})
            for s in range(500)]
    rows += [_ev(s, s % 7, "label_adopt", {"item_id": "v-1"}) for s in range(400)]
    rows += [_ev(250, 4, "partner_formed", {"other": 9})]
    rows += [_ev(430, 5, "council_elected", {"term": 1, "members": [1, 2, 3]})]
    feed = FR.build_feed(rows)
    ranked = sorted(feed["events"], key=lambda e: -e["sc"])
    assert ranked[0]["k"] == "council_elected"
    assert "partner_formed" in {e["k"] for e in ranked[:3]}
    # 配管 kind が上位を占拠していない
    assert ranked[0]["k"] not in ("belief_transmit", "label_adopt")


def test_score_is_deterministic():
    rows = [_ev(s, s % 4, "flyer_post", {"text": f"t{s}"}) for s in range(40)]
    rows += [_ev(9, 1, "group_found", {"name": "会", "group_id": 1})]
    a = FR.build_feed(rows)
    b = FR.build_feed(list(rows))
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


def test_first_occurrence_bonus_and_dedup_penalty():
    """同一 storyline の 1 件目 > 2 件目 > 3 件目(初出ボーナス+重複割引)。"""
    rows = [_ev(s * 3, 1, "label_adopt", {"item_id": "v-9"}) for s in range(6)]
    recs = FR.score_events(FR.collect(rows))
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True), "重複割引で単調に下がっていない"
    assert scores[0] > scores[1] > scores[2]
    assert recs[0]["parts"]["fst"] == 1.0 and recs[0]["parts"]["ded"] == 0.0
    assert recs[3]["parts"]["ded"] < 0.0


def test_magnitude_z_lifts_big_instances_within_kind():
    """kind 内の規模(reach)が大きいものほど上位(magnitude_z)。"""
    rows = [_ev(s, 2, "viral_cascade", {"post_id": s, "reach": s}) for s in range(1, 40)]
    recs = FR.score_events(FR.collect(rows))
    top = max(recs, key=lambda r: r["score"])
    assert top["_mag"] == 39.0


def test_pacing_is_subset_and_caps_are_reported():
    """ペーシングは掲載を絞るだけ(スコアは書き換えない)+ 落ちた件数を必ず出す。"""
    rows = []
    for s in range(200):
        rows.append(_ev(s, s % 6, "relation_tier", {"tier": 2, "other": (s + 1) % 6}))
        rows.append(_ev(s, s % 6, "label_adopt", {"item_id": f"v{s % 9}"}))
    rows += [_ev(50, 1, "group_found", {"name": "会", "group_id": 1})]
    feed = FR.build_feed(rows)
    st, cp = feed["stats"], feed["caps"]
    assert st["n_total"] == len(rows)
    assert st["n_kept"] < st["n_total"]
    assert cp["paced"] + cp["capped"] == st["n_total"] - st["n_kept"], "silent cap 禁止"
    # 掲載は時刻順
    assert [e["s"] for e in feed["events"]] == sorted(e["s"] for e in feed["events"])


def test_pacing_narrows_big_event_slots_after_a_burst():
    """大事件を連発させると、その直後だけ **大事件の掲載枠が狭まる**(L4D の緊張と緩和)。

    前半 = 大事件(importance 4)を密に、後半 = 同じ大事件を疎に置き、各行に記録された
    掲載枠 `pace` を比べる。静かな区間では枠が満枠(= 強度が抜けている)へ戻る。
    """
    burst = [_ev(s, s % 5, "venture_open", {"name": f"店{s}", "venture_id": f"o{s}"})
             for s in range(0, 40)]
    calm = [_ev(400 + s * 40, s % 5, "venture_open",
                {"name": f"店x{s}", "venture_id": f"x{s}"}) for s in range(0, 40)]
    recs = FR.score_events(FR.collect(burst + calm))
    FR.pace(recs, keep_ratio=1.0)                   # 順位の枠は満枠=ペーシングだけを見る
    n = len(recs)
    burst_slots = [r["pace"] for r in recs if r["step"] < 200]
    calm_slots = [r["pace"] for r in recs if r["step"] >= 200]
    assert min(burst_slots) < min(calm_slots), \
        "連発区間で大事件の枠が狭まっていない(ペーシングが効いていない)"
    assert max(calm_slots) == float(n), "静かな区間で枠が満枠へ戻っていない"


def test_pacing_never_exceeds_the_rank_budget():
    """掲載数は必ず順位の枠(= keep_ratio × 母集団)以下。スコア団子でも暴れない。

    dedup 上限でスコアが 1 点に潰れる kind(relation_tier / label_adopt)を大量に混ぜて、
    「閾値をわずかに動かすと団子が雪崩れ込む」旧方式の病理が再発しないことを固定する。
    """
    rows = []
    for s in range(200):
        rows.append(_ev(s, s % 6, "relation_tier", {"tier": 2, "other": (s + 1) % 6}))
        rows.append(_ev(s, s % 6, "label_adopt", {"item_id": f"v{s % 9}"}))
    recs = FR.score_events(FR.collect(rows))
    tied = max(sum(1 for r in recs if abs(r["score"] - x) < 1e-9) for x in
               {round(r["score"], 6) for r in recs})
    assert tied > 50, "同点の団子が出来ていない= 病理の再現テストになっていない"
    for ratio in (0.1, 0.25, 0.4):
        keep, drop = FR.pace(recs, keep_ratio=ratio)
        assert len(keep) <= -(-len(recs) * ratio // 1), f"枠 {ratio} を超えて掲載している"
        assert len(keep) + len(drop) == len(recs)


def test_top_importance_is_always_published():
    """S 層(importance 5)は連発の直後でも無条件掲載(memo §4-4)。"""
    rows = [_ev(s, s % 5, "venture_open", {"name": f"店{s}", "venture_id": f"o{s}"})
            for s in range(60)]
    rows += [_ev(30, -1, "disaster", {"kind": "台風", "phase": "onset"}, x=0.0, y=0.0)]
    feed = FR.build_feed(rows)
    assert any(e["k"] == "disaster" for e in feed["events"])
    dis = next(e for e in feed["events"] if e["k"] == "disaster")
    assert dis["p"] is False and dis["a"] == -1     # 世界イベントは位置なし


def test_empty_feed_for_plumbing_only_events():
    """レジストリに無い配管 kind だけのランはフィードが空(= 注入されない側へ落ちる)。"""
    rows = [_ev(s, s % 3, "transmission", {"item_id": "x"}) for s in range(50)]
    rows += [_ev(s, s % 3, "opinion_shift", {"old": 0.1, "new": 0.2}) for s in range(50)]
    feed = FR.build_feed(rows)
    assert feed["events"] == [] and feed["stats"]["n_total"] == 0


# ===========================================================================
# 3. 生成 HTML — 後方互換(バイト同一)と注入
# ===========================================================================
def _minimal_map(path: Path) -> None:
    city = {
        "buildings": [
            # L 字(bbox の右下 1/4 が欠けている)= 矩形散布との差が出る形
            {"id": "b1", "name": "テストビル", "kind": "retail", "levels": 6,
             "footprint": [[0, 0], [100, 0], [100, 50], [50, 50], [50, 100], [0, 100]]}],
        "nodes": [{"id": "n1", "x": 0, "y": 0, "name": "ノードA"},
                  {"id": "n2", "x": 100, "y": 0}],
        "edges": [{"klass": "primary", "layer": 0, "geometry": [[0, 0], [100, 0]]}],
        "pois": [], "railways": [], "meta": {"origin_latlon": [35.66, 139.70]},
    }
    path.write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")


def _write_run(tmp_path: Path, name: str, *, feed_events: bool,
               n_agents: int = 4, n_steps: int = 12) -> Path:
    """合成ラン。feed_events=False なら**レジストリ対象 kind を 1 件も含まない**。"""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    _minimal_map(map_path)
    (run_dir / "config.yaml").write_text(
        "world:\n"
        f"  map: {map_path.as_posix()}\n"
        "transit:\n"
        "  file: data/__no_transit_for_test__.json\n",
        encoding="utf-8")
    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
               "occupation": "会社員", "visitor": False,
               "has_bicycle": False, "has_car": False} for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    rows = []

    def add(step, aid, kind, payload, x=0.0, y=0.0):
        rows.append({"step": int(step), "agent_id": int(aid), "kind": kind,
                     "sim_min": 420 + int(step) * mv.STEP_MINUTES,
                     "x": float(x), "y": float(y),
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    for step in range(n_steps):
        for a in range(n_agents):
            add(step, a, "move_segment",
                {"mode": "walk", "pts": [[10.0 + a * 3, 0.0], [20.0 + a * 3, 0.0]]},
                x=10.0 + a * 3)
    add(2, 0, "enter_building", {"building": "b1", "floor": 2}, x=50.0, y=20.0)
    if feed_events:
        add(3, 1, "group_found", {"name": "つけめんの会", "group_id": 1,
                                  "purpose": "ゆるくつながる"}, x=30.0, y=10.0)
        add(5, 2, "label_coin", {"item_id": "v-1", "text": "フワき"}, x=40.0, y=12.0)
        add(7, -1, "disaster", {"kind": "台風", "phase": "onset"})
        for s in range(8, n_steps):
            add(s, 3, "label_adopt", {"item_id": "v-1", "text": "フワき"}, x=44.0)
    fields = [("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
              ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
              ("payload", pa.string()), ("rng_stream", pa.string()),
              ("llm_call_id", pa.string())]
    cols = {nm: [r[nm] for r in rows] for nm, _ in fields}
    pq.write_table(pa.table(cols, schema=pa.schema(fields)),
                   run_dir / "l1_events.parquet")
    return run_dir


def _gen(run_dir: Path, *flags: str) -> tuple[bytes, bytes]:
    r = subprocess.run([sys.executable, str(REPO_ROOT / "viz" / "make_viewer.py"),
                        str(run_dir), *flags],
                       cwd=REPO_ROOT, capture_output=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    return ((run_dir / "viewer.html").read_bytes(),
            (run_dir / "dashboard.html").read_bytes())


def test_no_new_template_tokens():
    """テンプレート(MAP_HTML/DASH_HTML)に新トークンを 1 つも足していない。

    足すと既存の「テンプレ無改変」テスト(test_org_ui / test_viewer_indoor)が崩れる。
    F1/F3 は既存の __COMMUNITY_JS__ / __INDOOR_JS__ へ追記合成するだけ。
    """
    both = mv.MAP_HTML + mv.DASH_HTML
    for tok in re.findall(r"__(?:FEED|SPOT|EVENT)[A-Z_]*__", both):
        pytest.fail(f"テンプレートに新トークン {tok} を足している(バイト同一が崩れる)")
    assert mv.MAP_HTML.count("__COMMUNITY_JS__") == 1
    assert mv.DASH_HTML.count("__COMMUNITY_JS__") == 0   # フィードは viewer だけ


def test_flags_off_is_byte_identical(tmp_path):
    """既定(フラグ未指定)= 従来出力。フィード対象イベントが在るランでも変わらない。"""
    rd = _write_run(tmp_path, "off_run", feed_events=True)
    a_v, a_d = _gen(rd)
    b_v, b_d = _gen(rd)
    assert a_v == b_v and a_d == b_d                    # 決定論(2 回生成で byte 一致)
    assert b"feedPane" not in a_v and b"SPOT_DENSITY_MIN" not in a_v
    assert b"evfeed" not in a_v


def test_old_run_without_feed_data_is_byte_identical_even_with_flag(tmp_path):
    """**旧ラン(該当データ無し)は --event-feed を明示しても viewer.html がバイト同一**。

    後方互換の合格条件(検収 a)。レジストリ対象 kind を 1 件も含まないランでは
    build_data が evfeed キーを足さず、注入文字列も 1 文字も増えない。
    """
    rd = _write_run(tmp_path, "old_run", feed_events=False)
    off_v, off_d = _gen(rd)
    on_v, on_d = _gen(rd, "--event-feed")
    assert on_v == off_v, "該当データ無しのランで viewer.html が変わった(後方互換違反)"
    assert on_d == off_d
    assert hashlib.sha256(on_v).hexdigest() == hashlib.sha256(off_v).hexdigest()


def test_event_feed_flag_injects_only_viewer(tmp_path):
    rd = _write_run(tmp_path, "feed_on", feed_events=True)
    off_v, off_d = _gen(rd)
    on_v, on_d = _gen(rd, "--event-feed")
    assert on_v != off_v
    assert b"feedPane" in on_v and b'id="fdList"' in on_v
    assert b"evfeed" in on_v
    # dashboard には右ペインの JS は入らない(__COMMUNITY_JS__ が MAP_HTML にしか無い)
    assert b"feedPane" not in on_d and b'id="fdList"' not in on_d


def test_indoor_v2_flag_changes_scatter_only_when_given(tmp_path):
    rd = _write_run(tmp_path, "spot_on", feed_events=False)
    off_v, off_d = _gen(rd)
    on_v, on_d = _gen(rd, "--indoor-v2")
    assert on_v != off_v and on_d != off_d              # 間取りは viewer/dashboard 両方に在る
    for probe in (b"SPOT_DENSITY_MIN", b"spotPointIn", b"spotInFP"):
        assert probe in on_v and probe not in off_v
    # sin 微動が消えている(上書き後の _agentSpot は nowT を使わない)
    assert b"amp*Math.sin(nowT" in off_v               # 旧実装は元テンプレに残っている
    assert on_v.count(b"amp*Math.sin(nowT") == off_v.count(b"amp*Math.sin(nowT")


def test_build_data_adds_no_key_without_flag(tmp_path):
    """build_data の既定引数では evfeed キーが増えない(payload バイト不変の根拠)。"""
    rd = _write_run(tmp_path, "bd_run", feed_events=True)
    assert "evfeed" not in mv.build_data(rd, include_traffic=False)
    data = mv.build_data(rd, include_traffic=False, event_feed=True)
    assert "evfeed" in data and data["evfeed"]["events"]


def test_integration_mock_run_feed_ranks_rare_over_plumbing(tmp_path):
    """統合: mock ラン(parquet)→ build_data → フィードが **希少イベントを上位**に置く。

    配管的に大量発生する kind(label_adopt / joint_invite / belief_transmit)を数百件、
    物語級の希少 kind(council_elected / partner_formed)を数件だけ混ぜたランを作り、
    ビューアに載る順位を確かめる(memo §2-3 の「配管:物語 = 1000:1」の選別)。
    """
    run_dir = tmp_path / "mock_feed"
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / "mock_feed_map.json"
    _minimal_map(map_path)
    (run_dir / "config.yaml").write_text(
        f"world:\n  map: {map_path.as_posix()}\n"
        "transit:\n  file: data/__no_transit_for_test__.json\n", encoding="utf-8")
    n_agents, n_steps = 8, 60
    (run_dir / "agents.json").write_text(json.dumps(
        [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "女",
          "occupation": "会社員", "visitor": False,
          "has_bicycle": False, "has_car": False} for i in range(n_agents)],
        ensure_ascii=False), encoding="utf-8")
    rows = []

    def add(step, aid, kind, payload, x=0.0, y=0.0):
        rows.append({"step": int(step), "agent_id": int(aid), "kind": kind,
                     "sim_min": 420 + int(step) * mv.STEP_MINUTES,
                     "x": float(x), "y": float(y),
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    for step in range(n_steps):
        for a in range(n_agents):
            add(step, a, "move_segment",
                {"mode": "walk", "pts": [[10.0 + a, 0.0], [12.0 + a, 0.0]]}, x=10.0 + a)
            # 配管(レジストリ外)+ 高頻度の普及系(レジストリ内・importance 2)
            add(step, a, "transmission", {"item_id": "v-1", "from": (a + 1) % n_agents})
            add(step, a, "label_adopt", {"item_id": "v-1", "text": "フワき"}, x=11.0 + a)
            if step % 2 == 0:
                add(step, a, "belief_transmit",
                    {"claim": "うわさ", "item_id": "b-1", "hop": 2}, x=11.0 + a)
    # 物語級(希少)
    add(17, 4, "partner_formed", {"other": 5}, x=30.0, y=8.0)
    add(41, -1, "council_elected", {"term": 1, "members": [0, 1, 2, 3, 4]})
    fields = [("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
              ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
              ("payload", pa.string()), ("rng_stream", pa.string()),
              ("llm_call_id", pa.string())]
    cols = {nm: [r[nm] for r in rows] for nm, _ in fields}
    pq.write_table(pa.table(cols, schema=pa.schema(fields)),
                   run_dir / "l1_events.parquet")

    data = mv.build_data(run_dir, include_traffic=False, event_feed=True)
    feed = data["evfeed"]
    ranked = sorted(feed["events"], key=lambda e: -e["sc"])
    assert ranked[0]["k"] == "council_elected", \
        f"最上位が議会改選でない: {[e['k'] for e in ranked[:3]]}"
    assert ranked[1]["k"] == "partner_formed"
    # 高頻度の配管系が上位 2 件を押しのけていない / レジストリ外は 1 件も載らない
    assert "transmission" not in feed["kinds"]
    plumbing = [e for e in feed["events"] if e["k"] in ("label_adopt", "belief_transmit")]
    assert max((e["sc"] for e in plumbing), default=-9e9) < ranked[1]["sc"]
    # 母集団の大半は落ちている(1000:1 の選別が起きている)
    assert feed["stats"]["n_kept"] < feed["stats"]["n_total"] * 0.5
    # ビューア HTML にもその順で載る
    on_v, _ = _gen(run_dir, "--event-feed")
    assert b"feedPane" in on_v and "議会改選".encode() in on_v


def test_feed_events_carry_seek_and_focus_material(tmp_path):
    """フィード行が「シーク+パン+当事者フォーカス」に必要な素材を全部持っている。"""
    rd = _write_run(tmp_path, "mat_run", feed_events=True)
    data = mv.build_data(rd, include_traffic=False, event_feed=True)
    evs = data["evfeed"]["events"]
    assert evs
    for e in evs:
        assert set(e) >= {"s", "m", "k", "a", "x", "y", "p", "t", "i", "sc", "g", "n"}
        assert 0 <= e["s"] < data["nSteps"]                    # シーク先が有効域
        assert e["k"] in data["evfeed"]["kinds"]               # ラベル/アイコンが引ける
    dis = next(e for e in evs if e["k"] == "disaster")
    assert dis["p"] is False                                   # 世界イベントはパンしない
    grp = next(e for e in evs if e["k"] == "group_found")
    assert grp["p"] is True and grp["a"] in data["ids"]        # 当事者をフォーカスできる


# ===========================================================================
# 4. F3 幾何 — quickjs で出荷 JS をそのまま実行して確かめる
# ===========================================================================
_JS_STUB = """
var window={}, document={getElementById:function(){return null;},
  createElement:function(){return {style:{}};}};
var devicePixelRatio=1, performance={now:function(){return 0;}};
function requestAnimationFrame(){return 0;} function cancelAnimationFrame(){}
var ctx={}, cv={}, cam={s:1,cx:0,cy:0}, D={ids:[],buildings:[],positions:[]};
function tf(x,y){return [x,y];}
function colorOf(){return '#fff';} function colOf(){return '#fff';}
function nameOf(a){return 'a'+a;} function themeAt(){return {k:0};}
var B={id:'b1', name:'L', cx:33.3, cy:66.6, levels:3, below:0,
       fp:[[0,0],[100,0],[100,50],[50,50],[50,100],[0,100]]};
"""
# 統合前(第136)の _agentSpot。対照実験に使う(新実装が本当に直しているかの反証材料)。
_OLD_SPOT = """
function oldSpot(b,f,id,lay,nowT){
  var z = lay.zones.length? lay.zones[_hash('a'+id)%lay.zones.length] : {r:lay.corridor};
  var rng=_rng(_hash('p'+id+':'+(b.id||b.name)+':'+f));
  var u=0.2+rng()*0.6, v=0.2+rng()*0.6;
  var x=z.r[0]+(z.r[2]-z.r[0])*u, y=z.r[1]+(z.r[3]-z.r[1])*v;
  var sp=(z.r[2]-z.r[0]); var amp=Math.min(1.4, Math.max(0.4, sp*0.06));
  x += amp*Math.sin(nowT*0.5+id*1.3); y += amp*Math.sin(nowT*0.42+id*2.1);
  return [x,y];
}
"""


def _spot_ctx():
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval(_JS_STUB)
    ctx.eval(mv._FLOOR_JS)
    ctx.eval(mv._SPOT_JS)
    ctx.eval(_OLD_SPOT)
    ctx.eval("var LAY=floorLayout(B,1);")
    return ctx


def test_f3_scatter_lands_inside_real_footprint():
    """新実装は L 字建物でも **全員が footprint の内側**。旧実装は外へ出る(対照)。"""
    ctx = _spot_ctx()
    out_new = ctx.eval("""(function(){var n=0;
      for(var id=0;id<400;id++){var p=_agentSpot(B,1,id,LAY,0.0);
        if(!spotInFP(B.fp,p[0],p[1])) n++;} return n;})()""")
    out_old = ctx.eval("""(function(){var n=0;
      for(var id=0;id<400;id++){var p=oldSpot(B,1,id,LAY,0.0);
        if(!spotInFP(B.fp,p[0],p[1])) n++;} return n;})()""")
    assert out_new == 0, f"新実装で {out_new}/400 人が建物の外に湧いている"
    assert out_old > 0, "対照(旧実装)が外に出ない= L 字の検証になっていない"


def test_f3_scatter_is_time_independent():
    """sin 微動の撤去: 位置は連続時刻 t に依存しない(= 静止した人は本当に静止する)。"""
    ctx = _spot_ctx()
    same = ctx.eval("""(function(){
      for(var id=0;id<400;id++){
        var a=_agentSpot(B,1,id,LAY,0.0), b=_agentSpot(B,1,id,LAY,987.65);
        if(a[0]!==b[0]||a[1]!==b[1]) return false; }
      return true;})()""")
    moved_old = ctx.eval("""(function(){var n=0;
      for(var id=0;id<400;id++){
        var a=oldSpot(B,1,id,LAY,0.0), b=oldSpot(B,1,id,LAY,987.65);
        if(a[0]!==b[0]||a[1]!==b[1]) n++; } return n;})()""")
    assert same is True, "新実装がまだ時刻に依存している(揺れが残っている)"
    assert moved_old > 0, "対照(旧実装)が揺れない= 揺れの検証になっていない"


def test_f3_scatter_is_deterministic_per_agent():
    """同じ (建物, 階, agent) は常に同じ点(決定論・乱数不使用)。"""
    ctx = _spot_ctx()
    ok = ctx.eval("""(function(){
      for(var id=0;id<200;id++){
        var a=_agentSpot(B,1,id,LAY,0.0), b=_agentSpot(B,1,id,LAY,0.0);
        if(a[0]!==b[0]||a[1]!==b[1]) return false; }
      return true;})()""")
    assert ok is True


def test_f3_density_constants_present_and_ordered():
    """密度切替の閾値が出荷 JS に在り、意味のある値である(定数の腐敗検知)。"""
    src = mv._SPOT_JS
    mn = int(re.search(r"SPOT_DENSITY_MIN\s*=\s*(\d+)", src).group(1))
    slack = float(re.search(r"SPOT_DENSITY_SLACK\s*=\s*([\d.]+)", src).group(1))
    assert mn >= 2 and slack > 1.0
    # 「人数が閾値以上 かつ 画面が狭い」の連言で切り替える(片方だけで切らない)
    assert "here.length>=SPOT_DENSITY_MIN" in src and "SPOT_DENSITY_SLACK" in src
