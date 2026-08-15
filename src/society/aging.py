"""加齢・誕生日(AGE-F。第116バッチ 2026-08-15・**既定 OFF**)。

正典: docs/plans/age-diversity-plan.md §4-8(AGE-F)。

塞ぐ穴(同書 §2-3): `agent.age` は `persona.py` で**一度だけ**書かれ、以後**一切変化しない**。
誕生日なし・加齢なし。10 日ランでは該当は約 0.07% なので現行ホライズンでは非問題だが、
「歳を取らない街」という状態そのものが台帳に残っていなかった。ここでは**最小**だけ入れる。

やること(これだけ):
  - 暦の実日付が個体の誕生日と一致した日に `agent.age += 1`。
  - 既存の `life_event`(payload["kind"] = "birthday")を 1 件記録する。**新しい L1 kind は足さない**
    (population.py の emigrate/settle/birth と同じ作法)。

**やらないこと**(計画どおり・本選後):
  - ライフステージ遷移(就学・成人・退職)。年齢帯を跨いだ瞬間に職業・世帯・就業を
    組み替える機構は**入れない**。入れるなら人口の内生化(population.py)と合流させる筋。
  - 年齢起因の死亡(死は health.severity 経路のまま)。

設計上の約束:
  - **乱数を 1 draw も引かない**。誕生日は `(seed, 安定 persona id)` の blake2b から
    決める純関数(friends._stable_uniform / ontology._stable_uniform と同方式)= プロセス跨ぎ・
    resume・別ランで同一。名簿に生年月日が無いための**正直な代理**であり、実在の
    出生日分布(1〜3 月にやや多い等)には合わせていない = 一様である旨を明記する。
  - **要 world.calendar.enabled**。暦が無いと「実日付」が無く、誕生日を定義できない。
    calendar OFF では完全 no-op(捏造した暦で誕生日を作らない)。
  - **1 個体 1 年 1 回**。日境界ガード(`sim._aging_day`)で同じ日を二度見ない。
  - **LLM 呼ゼロ増**・プロンプト不変・予算不変(R1)。
  - 既定 OFF では 1 行も通らない = 乱数消費・イベント列ともバイト一致(golden を守る)。
"""
from __future__ import annotations

import hashlib

from .observer.schema import Event

#: 既存 `life_event` の下位 kind(新しい L1 kind は足さない)。
KIND_BIRTHDAY = "birthday"

DEFAULTS: dict = {
    "enabled": False,
    "seed": 20260815,     # 誕生日の安定ハッシュ種(run.seed と独立=別ランでも同じ人は同じ誕生日)
    "max_age": 120,       # 上限(暴走防止。実在の最高齢を超えて数え上げない)
}


def build_cfg(raw) -> dict:
    """conf の `aging` ブロックを型強制つきで正準化する(既定 OFF)。"""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001
        pass
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", DEFAULTS["enabled"])),
        "seed": int(raw.get("seed", DEFAULTS["seed"])),
        "max_age": int(raw.get("max_age", DEFAULTS["max_age"])),
    }


def cfg_of(sim) -> dict:
    cfg = getattr(sim, "_agingcfg", None)
    if cfg is None:
        cfg = build_cfg((sim.cfg.get("aging", {}) or {}))
        sim._agingcfg = cfg
    return cfg


def _pid(agent) -> str:
    """安定 persona id(プールは pool_pid=run.seed 非依存で固定、直接ランは agent.id)。"""
    pid = getattr(agent, "pool_pid", None)
    return str(pid) if pid is not None else str(agent.id)


def birthday_of(agent, cfg: dict) -> tuple[int, int]:
    """個体の誕生日 (月, 日)。**決定論・乱数ゼロ**・プロセス跨ぎで安定。

    名簿に生年月日が無いので `(seed, persona id)` の安定ハッシュから 1〜365 日目を選ぶ。
    ★正直な限界: **一様分布**であって実在の出生日分布ではない。2/29 は選ばない
    (平年に誕生日が消える個体を作らないため = 365 日から選ぶ)。
    """
    h = hashlib.blake2b(f"{int(cfg['seed'])}\x1fbday\x1f{_pid(agent)}".encode("utf-8"),
                        digest_size=8).digest()
    doy = int.from_bytes(h, "big") % 365 + 1        # 1..365(平年の通日)
    import datetime
    d = datetime.date(2001, 1, 1) + datetime.timedelta(days=doy - 1)  # 2001 = 平年
    return d.month, d.day


def phase_day(sim, step: int, sim_min: int) -> None:
    """日次境界: 今日が誕生日の個体を 1 歳上げる(既定 OFF=即 return=バイト一致)。

    `_phase_calendar_weather` の**後**に置く(当日の実日付が確定してから読む)。
    在場者だけを見る(`sim.agents` = その日実体化されている面々)。
    """
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return
    cal = getattr(sim, "calendarcfg", None)
    if not cal or not cal.get("enabled"):
        return                                     # 暦が無ければ誕生日は定義できない(捏造しない)
    day = sim_min // 1440
    if day == getattr(sim, "_aging_day", -1):
        return                                     # 1 日 1 回(resume でも二度数えない)
    sim._aging_day = day
    from .world import calendar as _calendar
    today = _calendar.date_of(cal, sim_min)
    md = (today.month, today.day)
    cap = int(cfg["max_age"])
    for agent in sim.agents:
        if birthday_of(agent, cfg) != md:
            continue
        new_age = int(getattr(agent, "age", 0) or 0) + 1
        if new_age > cap:
            continue
        agent.age = new_age
        # 年齢から precompute された個体プロファイルを無効化する(歳を取ったのだから
        # 作り直すのが正しい)。AGE-C の思考係数と media の視聴プロファイルが対象。
        agent._age_cog = None
        agent._media_profile = None
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="life_event", x=agent.x, y=agent.y,
                             payload={"kind": KIND_BIRTHDAY, "age": new_age}))
