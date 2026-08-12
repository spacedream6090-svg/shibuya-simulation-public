"""``summary.json`` への観測タリー(``provenance``)配線の検証(第109バッチ D2)。

何を解く問題か
--------------
第105〜108 バッチで入った現実被覆レーン(健康 H1 / 都市運営 / 環境事件 H5 / 昇降設備 /
街路の生業 / 車内空間)は、いずれも ``provenance(sim)`` を持ちながら
**``summary.json`` へ配線されていなかった**。各 module が「書き出しは別レーンの所有」と
正直に書いていた状態で、その帰結として縦煙(ドライラン)では

  「その機能が **OFF だった** のか、**ON だが 1 件も起きなかった** のか」が
  L1 の kind を数えるだけでは区別できない

という判定不能が残っていた(第108 の申し送り)。本バッチはこれを配線だけで閉じる。

守るもの
--------
① **OFF ではキー自体を出さない**(= 既存ラン・golden の summary.json は無風)。
   キーを 0 で生やすのは「欠測を 0 と偽らない」という本リポジトリの規約に反する。
② ON では ``summary[key] == module.provenance(sim)``(観測量を summary 側で作り直さない)。
③ **キー順を壊さない**(既に ON で回している medical / lost_property などのランの
   summary.json のキー並びが動かないよう、新規キーは末尾に足す)。
④ ★**取りこぼし検知**: ``provenance(sim)`` を持つ module は、summary へ配線されているか、
   「別の出力先を持つ」と理由つきで宣言されているかのどちらかでなければならない
   (= 次に provenance を書いた人が配線を忘れられない)。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from society import city_ops as CO
from society import facility_devices as FD
from society import health as H
from society import incidents_env as IE
from society import street_life as SL
from society import transit_interior as TI
from society.config import load_config
from society.engine.simulation import Simulation

REPO = Path(__file__).resolve().parents[1]
SOCIETY = REPO / "src" / "society"
FINALIZE = SOCIETY / "engine" / "simulation.py"

#: 本バッチで配線した (summary のキー, module) の対。
WIRED = (("health", H), ("city_ops", CO), ("incidents_env", IE),
         ("facilities", FD), ("street_life", SL), ("transit_interior", TI))

#: 6 層を同時に ON にする dotlist(mock・小規模でも全キーが立つ最小構成)。
ON = {
    "health.enabled": "true",
    "health.severity.enabled": "true",       # provenance が None を返す条件は severity 側
    "city_ops.enabled": "true",
    "incidents_env.enabled": "true",
    "world.facilities.enabled": "true",
    "street_life.enabled": "true",
    "transit_interior.enabled": "true",      # ★provenance は pulse OFF でも「pulse_on: false」を返す
}

#: ``provenance(sim)`` を持つが **summary 以外**へ出す module と、その行き先(理由つき)。
#: ここに書かずに配線も無い module が現れたら ④ のテストが落ちる。
ELSEWHERE = {
    "cognition/calib.py":
        "run_manifest.json の cognition キー(observer/manifest.py が集約して書く)",
    "cognition/fire.py":
        "cognition/calib.provenance が集約する(単独の出力先を持たない)",
    "cognition/watch.py":
        "cognition/calib.provenance が集約する(単独の出力先を持たない)",
    "cognition/plasticity.py":
        "cognition/calib.provenance が集約する(単独の出力先を持たない)",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, **ov):
    dot = ["run.seed=42", "run.n_agents=15", "run.n_steps=24", f"run.name={name}",
           "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _summary(sim):
    return json.loads((sim.out_dir / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def off_summary(tmp_path_factory):
    sim = _sim(tmp_path_factory.mktemp("prov_off"), "prov_off")
    sim.run()
    return _summary(sim)


@pytest.fixture(scope="module")
def on_run(tmp_path_factory):
    sim = _sim(tmp_path_factory.mktemp("prov_on"), "prov_on", **ON)
    sim.run()
    return sim, _summary(sim)


# =========================================================================== #
# ① OFF ではキー自体を出さない
# =========================================================================== #
@pytest.mark.parametrize("key,mod", WIRED, ids=[k for k, _ in WIRED])
def test_summary_has_no_key_when_the_feature_is_off(off_summary, key, mod):
    assert key not in off_summary, \
        f"{key} が既定 OFF のランの summary に生えた(既存ラン・golden が無風でなくなる)"


def test_off_summary_still_has_the_pre_existing_keys(off_summary):
    """配線が既存キーを 1 つも消していない(足しただけであることの固定)。"""
    for key in ("n_agents", "n_steps", "n_events", "event_kinds", "llm_calls",
                "n_items", "n_transmissions", "total_adoptions", "out_dir", "files"):
        assert key in off_summary, key


# =========================================================================== #
# ② ON では module の provenance がそのまま載る
# =========================================================================== #
@pytest.mark.parametrize("key,mod", WIRED, ids=[k for k, _ in WIRED])
def test_summary_carries_the_module_provenance_when_on(on_run, key, mod):
    sim, summary = on_run
    prov = mod.provenance(sim)
    assert prov is not None, f"{key}: ON にしたのに provenance が None(ON セットの取り違え)"
    assert key in summary, f"{key} が summary に無い(配線が外れている)"
    assert summary[key] == prov, f"{key}: summary 側で観測量を作り直している"


def test_wired_keys_are_appended_after_the_pre_existing_ones(on_run):
    """③ 新規キーは**末尾**に足す(既に ON で回しているランのキー順を壊さない)。"""
    _, summary = on_run
    keys = list(summary)
    first_new = min(keys.index(k) for k, _ in WIRED if k in keys)
    assert keys.index("files") < first_new
    assert keys.index("elapsed_sec") < first_new


# =========================================================================== #
# ④ 取りこぼし検知(次に provenance を書いた人が配線を忘れられない)
# =========================================================================== #
def _modules_with_provenance() -> dict[str, Path]:
    """``def provenance(sim)`` を持つ module の {相対パス: 実パス}。"""
    got: dict[str, Path] = {}
    for path in sorted(SOCIETY.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "provenance" \
                    and node.args.args and node.args.args[0].arg == "sim":
                got[path.relative_to(SOCIETY).as_posix()] = path
    return got


def _modules_called_in_finalize() -> set[str]:
    """``engine/simulation.py`` が ``<module>.provenance(...)`` を呼んでいる module の相対パス。

    ★import されているだけでは配線とみなさない: 認知スタック(fire / watch / plasticity)は
      別の用途で import されているので、名前の出現を根拠にすると**配線されていないのに
      配線済みと誤判定する**(実測してそうなった)。呼び出しの形そのものを見る。
    """
    tree = ast.parse(FINALIZE.read_text(encoding="utf-8"))
    alias: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = (node.module or "").replace(".", "/")
            for name in node.names:
                rel = f"{base}/{name.name}.py" if base else f"{name.name}.py"
                alias[name.asname or name.name] = rel
    got: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "provenance"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in alias):
            got.add(alias[node.func.value.id])
    return got


def test_every_module_with_provenance_is_wired_or_declares_its_destination():
    mods = _modules_with_provenance()
    assert len(mods) >= 20, "provenance を持つ module の走査に失敗している"
    called = _modules_called_in_finalize()
    assert {"health.py", "city_ops.py", "incidents_env.py", "facility_devices.py",
            "street_life.py", "transit_interior.py"} <= called, \
        "本バッチで配線した 6 層の呼び出しが finalize から消えている"
    missing = sorted(rel for rel in mods
                     if rel not in called
                     and not (ELSEWHERE.get(rel) or "").strip())
    assert not missing, (
        f"provenance を持つのに summary へ配線されていない module: {missing}。"
        "engine/simulation.py の finalize へ 1 ブロック足すか、本テストの ELSEWHERE へ"
        "**行き先を理由つきで**宣言すること(黙って落とさない)")


def test_the_elsewhere_declarations_are_not_silently_wired_too():
    """ELSEWHERE の 4 件が「実は finalize からも呼ばれている」状態になっていないこと。

    (= 宣言が現実と食い違ったまま残るのを防ぐ。どちらか一方が正しい。)
    """
    called = _modules_called_in_finalize()
    assert not (set(ELSEWHERE) & called), \
        f"ELSEWHERE 宣言と実際の配線が二重になっている: {sorted(set(ELSEWHERE) & called)}"


def test_declared_elsewhere_modules_really_exist():
    """ELSEWHERE の宣言が古びていないこと(消えた module の言い訳を残さない)。"""
    mods = _modules_with_provenance()
    assert set(ELSEWHERE) <= set(mods), \
        f"存在しない module の宣言が残っている: {sorted(set(ELSEWHERE) - set(mods))}"
