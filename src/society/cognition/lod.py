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

import math

# ---- 入力解像度LOD(docs/plans/input-resolution-lod.md)------------------------
# mid = 現行の定数(poi[:3]・recent(4)・retrieve(n=3)・feed[:3]・人は全列挙)と完全一致
# = ON でも全員 mid なら注入内容は現行と同じ。people_n 0 / salience_k 0 は「現行のまま」。
# 下限側(narrow)を細かく振る(入力は増やす側の効果が飽和する=リサーチ §3.3)。
# scene_n = 構造化シーン記述(world/scene_desc)の記述件数。scene_desc OFF では誰も読まない
# =注入に影響しない(既定 OFF の scene seam と直交)。mid は現行相当=3。
INPUT_RES_DEFAULTS = {
    "enabled": False,
    "levels": {
        "narrow": {"share": 0.33, "poi_n": 1, "people_n": 2, "recent_n": 2,
                   "retrieve_n": 1, "feed_n": 1, "salience_k": 2, "scene_n": 2},
        "mid":    {"share": 0.34, "poi_n": 3, "people_n": 0, "recent_n": 4,
                   "retrieve_n": 3, "feed_n": 3, "salience_k": 0, "scene_n": 3},
        "wide":   {"share": 0.33, "poi_n": 5, "people_n": 0, "recent_n": 6,
                   "retrieve_n": 5, "feed_n": 5, "salience_k": 0, "scene_n": 4},
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


# ---- 二層予算(DPH-B。docs/plans/dayplan-horizon-plan.md §3.3・既定 OFF)------------
#
# 何を解く問題か
# --------------
# 現行の `LodBudget` は step あたりの単一カウンタで、`take()` を呼ぶのは scheduler の
# 3 箇所だけ(対面/割込みの確定発火・媒体/独り言の抽選当選・返答保証)。
# **朝の計画と夜の内省は予算の外**を無条件に走るので、
#   (a) 実効の per-step 上限が `cap + 予算外呼` になり(実測 cap 60 のランで 1 step 96 呼)、
#   (b) cap が binding なランでは発火側(先に走る)が食い切って **返答保証が枯渇**する
#       (実測: cap 60 で reply −96%・08-23 時は 1 件も返らない)。
#
# 契約(ON のとき)
# ----------------
# - **総量は増えない**: どのレーンから取っても `used <= max_per_step` を必ず満たす。
#   plan / reflect が予算の**中**へ入るので、ON の総呼数は OFF 以下にしかならない
#   (= 分配だけを変える。tests が機械固定する)。
# - **予約は必ず守られる**: general(発火・独り言・投稿)は自分の枠しか使えない。
#   一方 reply / life(plan・reflect)は自分の枠を使い切ったら general の**余り**を借りる。
# - **正直な限界**: 「未使用の reserved を同 step 内で general へ流す」(計画書 §3.3 の
#   最後の 1 行)は**実装しない**。run_step のフェーズ順が
#   planning(life)→ drive(general)→ decide(reply)→ reflect(life) なので、
#   general が走る時点では reply / reflect の予約が余るかどうかがまだ判らない。
#   先に流すと予約が守れず、後から流しても general は既に終わっている。
#   代わりに**逆向き**(予約超過は general の余りを借りる)だけを入れてある =
#   総量は cap のまま・予約は必ず守られる・遊ぶのは「予約が余った分」だけ。
#
# life_borrow(2026-08-25。正典 docs/plans/llm-budget-respec.md §4-3)
# --------------------------------------------------------------------
# 上の「逆向きだけ許す」設計には、実機で顕在化した副作用がある: plan の需要は
# **在場全員が毎朝 1 件**なので、cap が飽和するランでは life が general の余りを
# 毎 step 全量借り切り、**自発発火(general)が昼帯ゼロ**になる(24c 実測:
# step25 以降 n_fires=0・需要は 18.7 万件/step まで積み上がって全拒否)。
# `life_borrow: false` は life の借用**だけ**を止める(reply の借用は 1 行も変えない)。
# 溢れた plan は既存の FIFO 繰り越し → 骨格フォールバックの梯子が受ける。
# **既定 true = 従来の借用そのまま = 1 バイト同一**。
BUDGET_TIER_DEFAULTS = {
    "enabled": False,
    "reply_share": 0.20,      # 返答保証の先取り枠(cap に対する割合)
    "life_share": 0.30,       # 生活基盤(朝の計画 plan / 夜の内省 reflect)の先取り枠
    "max_defer_steps": 18,    # plan/reflect の FIFO 繰り越し上限(18 step = 3 時間 @Δt10)
    "life_borrow": True,      # life が general の余りを借りるか(false = 自発発火の保護)
}

# 観測ラベル(purpose)→ 予算レーン。**観測の粒度と配分の粒度を分ける**ための 1 枚。
#   観測は「どの呼び出し site で落ちたか」を purpose 別に数えたい(DPH-O ④)が、
#   配分は 3 レーンで足りる。既知でない purpose は general 扱い(捏造しない)。
PURPOSE_LANE: dict[str, str] = {
    "reply": "reply",
    "plan": "life", "reflect": "life",
    "face": "general", "interrupt": "general", "media": "general",
}
BUDGET_LANES: tuple[str, ...] = ("life", "reply", "general")


def build_budget_cfg(raw) -> dict | None:
    """config(lod.budget.tiers)→ 検証済み cfg。enabled=False なら None(=完全 no-op)。"""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001 (omegaconf 不在でも動く)
        pass
    raw = dict(raw or {})
    if not raw.get("enabled", False):
        return None
    cfg = dict(BUDGET_TIER_DEFAULTS)
    cfg.update({k: v for k, v in raw.items() if k in BUDGET_TIER_DEFAULTS})
    cfg["enabled"] = True
    cfg["reply_share"] = max(0.0, float(cfg["reply_share"]))
    cfg["life_share"] = max(0.0, float(cfg["life_share"]))
    cfg["max_defer_steps"] = max(0, int(cfg["max_defer_steps"]))
    cfg["life_borrow"] = bool(cfg["life_borrow"])
    if cfg["reply_share"] + cfg["life_share"] > 1.0:
        raise ValueError("lod.budget.tiers: reply_share + life_share は 1.0 以下。")
    return cfg


class LodBudget:
    """step あたりの LLM 発火予算。

    `tiers=None`(既定)では第30バッチ以来の単一カウンタと**完全に同一**に振る舞う
    (`take()` の返り値列も同じ)。`counters` を渡すと purpose 別の許可/拒否件数を
    そこへ積む(DPH-O ④ の観測。**判定には 1 度も使わない**)。
    """

    def __init__(self, max_per_step: int, tiers: dict | None = None,
                 counters: dict | None = None):
        self.max_per_step = int(max_per_step)
        self.used = 0
        self.tiers = tiers or None
        self.counters = counters                   # None = 観測 OFF(dict も作らない)
        self.caps: dict[str, int] = {}
        self.lane_used: dict[str, int] = {}
        # life が general の余りを借りてよいか(既定 True = 従来の挙動)。
        # 旧 cfg dict(キーを持たない)から組んでも既定へ落ちる = 後方互換。
        self.life_borrow = bool((self.tiers or {}).get("life_borrow", True))
        if self.tiers:
            cap = self.max_per_step
            # ★切り上げ: 宣言された予約が丸めでゼロへ消えない(cap が小さいランほど
            #   予約が要るのに、切り捨てだと cap=4 × 0.20 → 0 枠になって飢餓が直らない)。
            #   round(…, 9) は 300×0.2=60.000000000000004 のような float 誤差での
            #   繰り上げ事故を防ぐ(simulation.py の n_proportional と同じ作法)。
            reply = min(cap, math.ceil(round(cap * float(self.tiers["reply_share"]), 9)))
            life = min(cap - reply,
                       math.ceil(round(cap * float(self.tiers["life_share"]), 9)))
            self.caps = {"reply": int(reply), "life": int(life),
                         "general": max(0, cap - int(reply) - int(life))}
            self.lane_used = {k: 0 for k in BUDGET_LANES}

    # -- step 境界 ---------------------------------------------------------- #
    def reset(self) -> None:
        self.used = 0
        if self.tiers:
            self.lane_used = {k: 0 for k in BUDGET_LANES}

    # -- 消費 --------------------------------------------------------------- #
    def take(self, purpose: str | None = None) -> bool:
        ok = self._take(PURPOSE_LANE.get(str(purpose or ""), "general"))
        if self.counters is not None:              # 観測(判定に影響しない)
            row = self.counters.setdefault(str(purpose or "unknown"),
                                           {"granted": 0, "denied": 0})
            row["granted" if ok else "denied"] += 1
        return ok

    def _take(self, lane: str) -> bool:
        if self.used >= self.max_per_step:         # ★総量の硬い上限(ON/OFF 共通)
            return False
        if not self.tiers:
            self.used += 1
            return True
        if self.lane_used[lane] < self.caps[lane]:
            self.lane_used[lane] += 1
            self.used += 1
            return True
        # 予約レーン(reply / life)は general の**余り**だけを借りられる。
        # general 自身は借りない = 予約が先に食われることが原理的に起きない。
        # ★life_borrow=False では life(朝の計画 / 夜の内省)の借用だけを止める =
        #   自発発火(general)の枠が plan 需要に食い尽くされない。reply の借用は
        #   1 行も変えない(返答保証はレーン設計の第一目的)。
        if lane == "life" and not self.life_borrow:
            return False
        if lane != "general" and self.lane_used["general"] < self.caps["general"]:
            self.lane_used["general"] += 1
            self.used += 1
            return True
        return False
