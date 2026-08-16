"""B10(第122バッチ 2026-08-16): 手順書の縮小ランコマンドは **present_cap を対指定**する。

なぜ機械検査するか
------------------
`pool.enabled: true` のプロファイル(本選 conf)では **在場人口は `pool.present_cap` 側が効く**。
`run.n_agents` は名目 N でしかないので、`run.n_agents=2000` だけを渡して縮小したつもりで走らせると
**present_cap=250,000 のまま 25 万体ぶんの在場を組み立てる**。2026-08-16 に実地で
「CPU 高負荷・RSS 2.5GB・GPU 使用率 0%・run dir が空のまま」という事故が起きている
(pre-production-gate.md B4)。手順書のコマンドを人間が読んで気づく話ではないので、
**文書側をテストで固定する**(手順書は運用の実行可能仕様だ、という扱い)。

検査規則(誤検知を避けるため意図的に狭い)
------------------------------------------
対象 = 下の 4 文書の中で、``\\`` の行継続を畳んだうえで
  ① ``scripts/run.py`` を含み(= 実行コマンド。説明文中の ``run.n_agents`` 断片は拾わない)
  ② ``run.n_agents=N`` と ``run.n_steps=`` の**両方**を持つ
論理行。この 2 条件を満たす行は ``pool.present_cap=N``(**n_agents と同値**)を持たねばならない。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCS = (
    "docs/plans/decision-dashboard.md",
    "ops/runbook-first-night.md",
    "ops/finals-compute-checklist.md",
    "docs/plans/pre-production-gate.md",
)

_N_AGENTS = re.compile(r"run\.n_agents=(\d[\d_]*)")
_N_STEPS = re.compile(r"run\.n_steps=")
_PRESENT_CAP = re.compile(r"pool\.present_cap=(\d[\d_]*)")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """``\\`` 継続を畳んで (先頭行番号, 論理行) の列にする。"""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not buf:
            start = i
        if line.endswith("\\"):
            buf.append(line[:-1])
            continue
        buf.append(line)
        out.append((start, " ".join(buf)))
        buf = []
    if buf:
        out.append((start, " ".join(buf)))
    return out


def _run_commands(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    return [(n, s) for n, s in _logical_lines(text) if "scripts/run.py" in s]


@pytest.mark.parametrize("rel", DOCS)
def test_scaled_run_commands_pair_present_cap_with_n_agents(rel):
    """縮小ランのコマンドは present_cap を n_agents と**同値で**対指定している。"""
    path = REPO_ROOT / rel
    assert path.exists(), f"手順書が見つからない: {rel}"
    problems: list[str] = []
    for lineno, cmd in _run_commands(path):
        m = _N_AGENTS.search(cmd)
        if not m or not _N_STEPS.search(cmd):
            continue                       # 実行コマンドでない/規模指定のない行は対象外
        n_agents = int(m.group(1).replace("_", ""))
        cap = _PRESENT_CAP.search(cmd)
        if cap is None:
            problems.append(
                f"{rel}:{lineno} run.n_agents={n_agents} に pool.present_cap が無い"
                f"(pool ON では在場は present_cap 側が効く) -> {cmd.strip()[:160]}")
        elif int(cap.group(1).replace("_", "")) != n_agents:
            problems.append(
                f"{rel}:{lineno} run.n_agents={n_agents} と "
                f"pool.present_cap={cap.group(1)} が不一致 -> {cmd.strip()[:160]}")
    assert not problems, "\n".join(problems)


def test_checker_would_catch_a_missing_present_cap():
    """検査そのものの反証(規則を緩めたら気づけるようにする自己テスト)。"""
    bad = ("python scripts/run.py --profile conf/finals_observe.yaml \\\n"
           "  run.seed=42 run.n_agents=2000 run.n_steps=288 run.name=x\n")
    (lineno, cmd), = [(n, s) for n, s in _logical_lines(bad)
                      if "scripts/run.py" in s]
    assert lineno == 1
    assert _N_AGENTS.search(cmd) and _N_STEPS.search(cmd)
    assert _PRESENT_CAP.search(cmd) is None          # ← これが「落ちるべき形」

    good = bad.replace("run.n_steps=288", "run.n_steps=288 pool.present_cap=2000")
    (_, cmd2), = [(n, s) for n, s in _logical_lines(good)
                  if "scripts/run.py" in s]
    cap = _PRESENT_CAP.search(cmd2)
    assert cap and int(cap.group(1)) == int(_N_AGENTS.search(cmd2).group(1))
