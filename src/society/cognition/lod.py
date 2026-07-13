"""LOD の家 — 予算(1step あたりの LLM 発火上限)+ 軸ごとの個体 tier 割当(第30バッチ)。

発火の「きっかけ」判定は v1 の surprise_of(即時トリガー)から
欲求駆動発火(drive.py + scheduler._phase_drive)へ移行した(Phase A, 2026-07-04)。
本モジュールは (1) 予算(インフラ上限 N≈90-480 decision/step)と、
(2) **LOD 軸の共通割当機構**(第30バッチ)を担う。

LOD 軸の一貫設計(ユーザー要件 2026-07-14):
- 入力解像度(input_res=「世界の見え方」の個体差)が最初の消費者。将来のモデル級 LOD
  (multi-model-lod M3)も **同じ assign_axis を使う** = LOD の割当設計を変えるときは
  ここ1箇所を変えれば全軸に波及する(一貫性)。
- 軸ごとに**独立の決定論 stream**("lod_<axis>")で割り当てる = 軸間は既定で無相関
  (直交実験のため)。相関させたい実験だけ同じ axis 名を共有させる。
- 割当は trait 非依存・k 非依存・初期化時1回固定(生得性の裏口と R1 を守る。
  docs/plans/input-resolution-lod.md §2)。
- 各軸は config 1キー(enabled)で完全に切り離せる(OFF=属性なし=バイト一致)。
"""
from __future__ import annotations

# ---- 入力解像度LOD(docs/plans/input-resolution-lod.md)------------------------
# mid = 現行の定数(poi[:3]・recent(4)・retrieve(n=3)・feed[:3]・人は全列挙)と完全一致
# = ON でも全員 mid なら注入内容は現行と同じ。people_n 0 / salience_k 0 は「現行のまま」。
# 下限側(narrow)を細かく振る(入力は増やす側の効果が飽和する=リサーチ §3.3)。
INPUT_RES_DEFAULTS = {
    "enabled": False,
    "levels": {
        "narrow": {"share": 0.33, "poi_n": 1, "people_n": 2, "recent_n": 2,
                   "retrieve_n": 1, "feed_n": 1, "salience_k": 2},
        "mid":    {"share": 0.34, "poi_n": 3, "people_n": 0, "recent_n": 4,
                   "retrieve_n": 3, "feed_n": 3, "salience_k": 0},
        "wide":   {"share": 0.33, "poi_n": 5, "people_n": 0, "recent_n": 6,
                   "retrieve_n": 5, "feed_n": 5, "salience_k": 0},
    },
}


def build_input_res_cfg(raw: dict | None) -> dict | None:
    """config(lod.input_res)→ 検証済み cfg。enabled=False なら None(=完全 no-op)。"""
    if not raw or not raw.get("enabled", False):
        return None
    levels_raw = raw.get("levels") or INPUT_RES_DEFAULTS["levels"]
    levels: dict[str, dict] = {}
    for name, spec in dict(levels_raw).items():
        base = dict(INPUT_RES_DEFAULTS["levels"].get(name, {"share": 0.0}))
        base.update({k: (float(v) if k == "share" else int(v))
                     for k, v in dict(spec).items()})
        levels[str(name)] = base
    if not levels:
        raise ValueError("lod.input_res.levels が空。")
    return {"enabled": True, "levels": levels}


def assign_axis(agent_ids: list[int], stream, levels: dict[str, dict]) -> dict[int, str]:
    """LOD 軸の決定論割当(全軸共通の機構)。

    stream = 軸専用の RngHub stream("lod_<axis>")。agent_id の昇順に一様乱数を1個ずつ
    引き、share の累積区間で水準を決める = trait 非依存・k 非依存・seed 再現。
    """
    names = list(levels.keys())
    shares = [max(0.0, float(levels[n].get("share", 0.0))) for n in names]
    total = sum(shares) or 1.0
    cum, acc = [], 0.0
    for s in shares:
        acc += s / total
        cum.append(acc)
    out: dict[int, str] = {}
    for aid in sorted(agent_ids):
        u = float(stream.random())
        for name, edge in zip(names, cum):
            if u <= edge:
                out[aid] = name
                break
        else:
            out[aid] = names[-1]
    return out


class LodBudget:
    def __init__(self, max_per_step: int):
        self.max_per_step = int(max_per_step)
        self.used = 0

    def reset(self) -> None:
        self.used = 0

    def take(self) -> bool:
        if self.used < self.max_per_step:
            self.used += 1
            return True
        return False
