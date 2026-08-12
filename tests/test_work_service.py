"""L2 域内従業者の「業務の実体」(work.service。既定 OFF)のテスト。

設計: docs/research/l2-work-reality.md。R1 の鉄則を継承:
- OFF(既定): L1 が純粋既定と完全一致(バイト一致)・serve/org_output 0 件・実行時状態も増えない。
- ON: (a)客の消費と同一 work_node の勤務中スタッフに serve が帰属(合成配置)。
      (b)スタッフ不在の消費は agent_id=-1 の記録(unstaffed)。
      (c)オフィス系の org_output が出勤者数に応じる。
      (d)LLM 呼数が OFF と同一(追加呼ゼロ=R1)。(e)同 seed 2回で L1 一致(決定論)。
      (f)serve/org_output が schema 登録済み。(g)接客要約が S2 ダイジェストに載る。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤24 step)。乱数不使用・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json
from pathlib import Path

from society import work as work_mod
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS, Event

_REPO = Path(__file__).resolve().parents[1]

_ON = {"work.service.enabled": "true"}
_OFF = {"work.service.enabled": "false"}
_ISL = {"prompts.interstitial.enabled": "true"}


def _sim(tmp_path, name, n=25, steps=24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# ------------------------------------------------------ (schema) 種別登録
def test_schema_registered():
    assert "serve" in EVENT_KINDS and "org_output" in EVENT_KINDS


# ------------------------------------------------------ OFF: 純粋既定と L1 完全一致
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。OFF では serve/org_output も業務状態も生えない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(work.service seam が no-op でない)"
    assert not [e for e in off.logger.events if e.kind in ("serve", "org_output")]
    for a in pure.agents:
        assert not hasattr(a, "_work_serve_by_label"), "OFF なのに業務アキュムレータが生えている"


# ------------------------------------------------------ (a) 接客: 同一ノードの勤務中スタッフへ帰属
def test_serve_attributed_to_on_duty_staff(tmp_path):
    """客の spend(飲食)と同一 work_node に在場・勤務中のスタッフに serve が帰属する(合成配置)。"""
    sim = _sim(tmp_path, "serve", **_ON)
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    customer, staff = sim.agents[0], sim.agents[1]
    # 全員を非スタッフ化(work 窓を閉じる)してから、staff だけを勤務中に据える(取り違え排除)
    for a in sim.agents:
        a.work_start_min = -1
    customer.node = node
    customer.x, customer.y = x, y
    staff.node = staff.work_node = node
    staff.x, staff.y = x, y
    staff.work_start_min, staff.work_end_min = 0, 1440   # 常時勤務窓

    since = len(sim.logger.events)
    sim.logger.log(Event(step=5, sim_min=700, agent_id=customer.id, kind="spend",
                         x=x, y=y, payload={"amount": 900.0, "balance": 9100.0,
                                            "cat": "food"}))
    scheduler._phase_work_service(sim, 5, 700, since)

    serves = [e for e in sim.logger.events if e.kind == "serve"]
    assert len(serves) == 1, f"serve が1件でない: {len(serves)}"
    s = serves[0]
    assert s.agent_id == staff.id and s.payload["cat"] == "food"
    assert s.payload["label"] == sim.workcfg["serve_by_cat"]["food"]
    assert s.payload["customer"] == customer.id and s.payload["node"] == node
    # 客側の既存イベント(spend)は不変(新イベントを足すだけ)
    assert sum(1 for e in sim.logger.events if e.kind == "spend") == 1


# ------------------------------------------------------ (b) 不在: agent_id=-1 の unstaffed 記録
def test_unstaffed_consumption_recorded(tmp_path):
    """スタッフ不在の消費は agent_id=-1 の serve{unstaffed:true} として記録される(挙動変更なし)。"""
    sim = _sim(tmp_path, "unstaffed", **_ON)
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    for a in sim.agents:                     # 誰も勤務中スタッフにならないよう窓を閉じる
        a.work_start_min = -1
    customer = sim.agents[0]
    customer.node = node
    customer.x, customer.y = x, y

    since = len(sim.logger.events)
    sim.logger.log(Event(step=3, sim_min=700, agent_id=customer.id, kind="spend",
                         x=x, y=y, payload={"amount": 900.0, "balance": 9100.0,
                                            "cat": "food"}))
    scheduler._phase_work_service(sim, 3, 700, since)
    serves = [e for e in sim.logger.events if e.kind == "serve"]
    assert len(serves) == 1 and serves[0].agent_id == -1
    assert serves[0].payload.get("unstaffed") is True


def test_unstaffed_suppressed_when_disabled(tmp_path):
    """record_unstaffed=false なら不在の消費で serve を1件も出さない。"""
    sim = _sim(tmp_path, "nounstaffed", **{**_ON, "work.service.record_unstaffed": "false"})
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    for a in sim.agents:
        a.work_start_min = -1
    customer = sim.agents[0]
    customer.node = node
    customer.x, customer.y = x, y
    since = len(sim.logger.events)
    sim.logger.log(Event(step=3, sim_min=700, agent_id=customer.id, kind="spend",
                         x=x, y=y, payload={"cat": "food"}))
    scheduler._phase_work_service(sim, 3, 700, since)
    assert not [e for e in sim.logger.events if e.kind == "serve"]


# ------------------------------------------------------ 接客対象外(taxi 等)は帰属しない
def test_non_serve_category_ignored(tmp_path):
    """serve_by_cat に無いカテゴリ(taxi)は接客対象外=serve を1件も出さない。"""
    sim = _sim(tmp_path, "taxi", **_ON)
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    customer, staff = sim.agents[0], sim.agents[1]
    for a in sim.agents:
        a.work_start_min = -1
    customer.node = node
    customer.x, customer.y = x, y
    staff.node = staff.work_node = node
    staff.work_start_min, staff.work_end_min = 0, 1440
    since = len(sim.logger.events)
    sim.logger.log(Event(step=5, sim_min=700, agent_id=customer.id, kind="spend",
                         x=x, y=y, payload={"cat": "taxi"}))
    scheduler._phase_work_service(sim, 5, 700, since)
    assert not [e for e in sim.logger.events if e.kind == "serve"]


# ------------------------------------------------------ (c) オフィス産出が出勤者数に応じる
def _fire_office(tmp_path, name, k):
    """k 人のオフィス出勤者を1職場ノードに据えて org_output を1回発火し、当該ノードの payload を返す。"""
    sim = _sim(tmp_path, name, n=max(k + 2, 8), **_ON)
    node = sim.city.pois_by_cat("office")[0]["node"]
    for a in sim.agents:                     # 全員リセット(既定の office 配属を排除=当該ノードだけ数える)
        a.work_node = ""
        a.work_start_min = -1
    for a in sim.agents[:k]:
        a.work_node = node
        a.work_building = ""                 # 職場単位キー = node
        a.work_start_min, a.work_end_min = 540, 1080
    sim._work_day = -1
    scheduler._work_office_output(sim, 0, 0, sim.workcfg)
    ours = [e for e in sim.logger.events
            if e.kind == "org_output" and e.payload["org"] == node]
    assert len(ours) == 1, f"org_output が当該ノードで1件でない: {len(ours)}"
    return ours[0].payload


def test_org_output_scales_with_attendance(tmp_path):
    """org_output の出勤者数 n と産出 output(=base_weight×人数)が出勤者数に比例する。"""
    p3 = _fire_office(tmp_path, "off3", 3)
    p5 = _fire_office(tmp_path, "off5", 5)
    assert p3["n"] == 3 and p3["output"] == 3.0 and p3["kind"] == "office"
    assert p5["n"] == 5 and p5["output"] == 5.0
    assert p5["output"] > p3["output"], "出勤者数が増えても産出が増えていない"


def test_org_output_once_per_day(tmp_path):
    """org_output は日次1回(同一日で二度目の呼び出しは何も出さない)。"""
    sim = _sim(tmp_path, "once", **_ON)
    node = sim.city.pois_by_cat("office")[0]["node"]
    for a in sim.agents:
        a.work_node = ""
        a.work_start_min = -1
    for a in sim.agents[:3]:
        a.work_node = node
        a.work_start_min, a.work_end_min = 540, 1080
    sim._work_day = -1
    scheduler._work_office_output(sim, 0, 0, sim.workcfg)
    n1 = len([e for e in sim.logger.events if e.kind == "org_output"])
    scheduler._work_office_output(sim, 1, 10, sim.workcfg)   # 同一日(day=0)
    n2 = len([e for e in sim.logger.events if e.kind == "org_output"])
    assert n1 > 0 and n2 == n1, "同一日で org_output が二重発火している"


# ------------------------------------------------------ (d) LLM 呼数が OFF と同一(追加呼ゼロ)
class _FixedLLM:
    """内容非依存の固定応答 backend。呼数だけを数える(下流分岐を無効化して呼数の純測定)。"""

    def __init__(self):
        self.response = json.dumps({"action": "speak", "text": "やあ"},
                                   ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, enabled):
    # interstitial も ON にして「業務ダイジェスト注入」経路まで動かす(それでも呼数は不変のはず)
    sim = _sim(tmp_path, name, **{"work.service.enabled": enabled,
                                  "prompts.interstitial.enabled": "true"})
    sim.llm = _FixedLLM()
    sim.run()
    return sim


def test_llm_call_count_invariant(tmp_path):
    """work.service ON/OFF で LLM 呼数が完全一致(追加 LLM 呼ゼロ=R1)。"""
    on = _run_fixed(tmp_path, "cc_on", "true")
    off = _run_fixed(tmp_path, "cc_off", "false")
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"work.service で呼数が変化(R1 違反): on={on.llm.calls} off={off.llm.calls}"


# ------------------------------------------------------ (e) 決定論(同 seed 2回で完全一致)
def test_on_deterministic(tmp_path):
    """work.service ON 同士 2 回で L1 完全一致(決定論・乱数不使用・mock 24 step)。"""
    a = _sim(tmp_path, "det_a", **{**_ON, **_ISL})
    a.run()
    b = _sim(tmp_path, "det_b", **{**_ON, **_ISL})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# ------------------------------------------------------ (f) 実ラン: org_output が発火する
def test_integration_org_output_fires(tmp_path):
    """既定名簿の office 系従業者から org_output が実ランで発火する(会社が『作っている』観測)。"""
    sim = _sim(tmp_path, "integ", n=40, steps=2, **_ON)
    sim.run()
    outs = [e for e in sim.logger.events if e.kind == "org_output"]
    assert outs, "実ランで org_output が1件も出ていない(office 系従業者が居ない?)"
    for e in outs:
        assert e.payload["n"] >= 1 and e.payload["output"] >= 1.0


# ------------------------------------------------------ (g) 接客要約が S2 ダイジェストに載る
def test_serve_digest_line():
    """業務アキュムレータからダイジェスト1行を客観記述で組む(空なら None=バイト一致)。"""

    class _A:
        pass

    a = _A()
    assert work_mod.digest_line(a) is None            # 属性不在=1行も足さない
    work_mod.note_serve(a, "飲食の接客")
    work_mod.note_serve(a, "飲食の接客")
    line = work_mod.digest_line(a)
    assert line is not None and "飲食の接客" in line and "2件" in line
    work_mod.clear_digest(a)
    assert work_mod.digest_line(a) is None            # 発火後は仕切り直し


# ------------------------------------------------------ (h) conf の接客写像に service 行があるか
# 第109バッチ D2 の小粒: services.spend_cat の既定は "service" で、受給の _spend は
# cat="service" で出る(conf/config.yaml が「work.serve_by_cat と対の seam」と明記)。
# ところが build_cfg は「conf が serve_by_cat を書いていればそちらが**丸ごと**勝つ」
# (`serve = dict(serve) if serve else dict(DEFAULT_SERVE_BY_CAT)`)ので、基底 conf に
# service 行が無いかぎり DEFAULT_SERVE_BY_CAT の service 行は 1 度も使われない。
# その結果、センサス較正台帳の service 職場 1,972 社が**構造的に無人接客**になる。
def test_default_map_has_the_service_row_but_the_base_conf_overrides_it():
    """既定写像には service 行がある / 基底 conf はそれを上書きして 4 行にする(現状の固定)。"""
    assert work_mod.DEFAULT_SERVE_BY_CAT["service"] == "窓口・施術の接客"
    base = work_mod.build_cfg(load_config().work)
    assert set(base["serve_by_cat"]) == {"food", "cafe", "nightlife", "shop"}, \
        "基底 conf の serve_by_cat が変わった(既定を触らない約束の確認)"
    # 受給側の cat(services.spend_cat)と対になる語がこれである、という seam の宣言
    assert str(load_config().services.spend_cat) == "service"


def test_finals_profile_restores_the_service_row():
    """本選 ONセット候補(conf/finals_observe.yaml)では service 行が復活している。"""
    cfg = load_config(profile=str(_REPO / "conf" / "finals_observe.yaml"))
    got = work_mod.build_cfg(cfg.work)["serve_by_cat"]
    assert got == work_mod.DEFAULT_SERVE_BY_CAT, \
        "finals プロファイルの serve_by_cat が work.py の既定写像と一致しない"
    assert got["service"] == "窓口・施術の接客"
