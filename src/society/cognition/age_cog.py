"""年齢 → 思考/推論の量(AGE-C。第116バッチ 2026-08-15・**既定 OFF**)。

正典: docs/plans/age-diversity-plan.md §4-4(AGE-C)。

塞ぐ穴(同書 §0 / §2-1 の全数実測): 既定 conf で年齢が影響するのは
**プロンプト中の文字列 `{age}歳` だけ**で、LLM 呼の割当・層・予算・内省頻度・
day_plan の全選抜が年齢盲目だった。つまり「9 歳と 80 歳が同じ思考周期・同じ発火閾値で
回っており、違いは自己紹介文の数字だけ」。しかもプロンプトでの年齢 steering は
**3 つの独立研究で無効が実証済み**(Santurkar ICML 2023 / Liu EMNLP 2024 /
Hong & Choi *The Gerontologist* 66(2) = モデルの 60/70/80/90 歳は互いに
コサイン類似度 0.73-0.90 でほぼ同一)= 現状は「効かないと分かっている方法」だった。
年齢を効かせるならアーキテクチャ側(個体の認知パラメタ)に置くほかない
(最も近い先行 ChildSafe も発達段階を**ハイパーパラメタ制約**として符号化している)。

**越えてはいけない線**(同書 §4-1 / conf/config.yaml の `mind.tiers.high.select: uniform`
に書かれた k* 自壊回避の掟):

  ✗ 年齢で層・モデル・予算・発火順を割り当てる(= 世界のアルゴリズムが量を決める)
  ✓ 年齢が個体の drive 分布を変え、**発火は本人の drive が決める**(量は行動から創発)

したがって本 module が触るのは次の 2 点**だけ**で、`LodBudget.take()` と
`requesters.sort(key=(-drive, id))` は 1 行も触らない:

  - `cognition/fire.py::_period_min` … 基本周期に個体倍率を掛ける(1 行)
  - `engine/scheduler.py::_eff_thr`  … 実効閾値へ不透明な delta を足す(1 行。
    `affect.threshold_delta` / `health.fatigue_threshold_delta` と同じ合流点)

レイヤ分担(no-fingerprint 契約 design §11):
  - `factors/registry.age_cognition_params` … 年齢 → 係数の**写像**(年齢という属性を読む層)
  - 本 module … conf の正準化 + agent へのキャッシュ(**不透明な 2 つの数**しか外へ出さない)
  - cognition/engine … 返ってきた数しか見ない

R1:
  - **既定 OFF**(`cognition.age.enabled: false`)= `period_mult` を 1 度も掛けず
    `theta_delta` を 1 度も足さない = 現行とバイト一致(golden を守る)。
  - **乱数ゼロ**: `age_cognition_params` は純関数(media.profile_for と同じ零 draw 型)。
    したがって ON にしても draw 順は 1 粒も動かない = `resume == straight` は無風。
  - **呼数**: cap が binding するラン(本選構成)では総数が cap 固定なので**増分ゼロ**で、
    変わるのは *誰が* 撃つかだけ。cap 非拘束の小規模構成では動きうるので、係数は
    `ref_age` で恒等になるよう正規化して両側へ対称に振れるようにしてある(§4-2)。
  - **k 非依存**: 入力は年齢と conf 定数だけ。belief 書き戻し・k.writeback を参照しない。
"""
from __future__ import annotations

from ..factors import registry as _reg

DEFAULTS: dict = {
    "enabled": False,
    # この年齢で period_mult=1.0 / theta_delta=0.0(= 現行較正の基準点)。
    # プール 100 万人の平均年齢は 36.17 歳(age-diversity-plan §1-1)。
    "ref_age": 38.0,
    # ---- 熟慮時間の年齢曲線(周期)----
    # Queen ら 2012 *Psych Aging* 27(4)(若 47 / 中 46 / 高 42 名): 1 セルあたりの
    # 処理時間 1,084 → 1,418 → 1,773 ms = **若→高で +63%**。3 群の代表年齢に knot を置く。
    # ★同研究は「開いたセルの割合は .78/.85/.85 で落ちない」とも報告しており、
    #   **閲覧幅(input_res)を年齢で削るのは根拠が弱い**ので本レーンではやらない。
    "period_knots": [[20.0, 1.00], [45.0, 1.31], [70.0, 1.64]],
    "period_gain": 1.0,      # 曲線の効き幅(0 = 恒等)
    "period_min": 0.7,
    "period_max": 1.8,
    # ---- 発火閾値の年齢曲線(周期とは**別の**曲線。単一の年齢係数は文献的に誤り)----
    # Steinberg ら 2018 *Dev Science* 21(2)(11 カ国 5,000 人超): 青年期のリスクは
    # 2 本の別曲線の差で、**感覚探求はピーク 19 歳**で以後低下・**自己制御は 23-26 歳で台地**。
    # ⇒ 10 代後半は「撃ちたくなりやすい」(閾値↓)・成人以降はゆるやかに戻り、
    #   高齢側は処理速度の低下(Salthouse 1996/2009)ぶんだけ僅かに上げる。
    # ★大きさは意図的に小さい(閾値の定義域は [0.30, 0.85])。符号+桁が文献・微調整は conf。
    "theta_knots": [[10.0, -0.020], [19.0, -0.060], [26.0, -0.020],
                    [40.0, 0.000], [65.0, 0.015], [82.0, 0.030]],
    "theta_gain": 1.0,
    "theta_clip": 0.08,      # delta の絶対値上限(暴走防止)
}

_BOOL_KEYS = ("enabled",)
_FLOAT_KEYS = ("ref_age", "period_gain", "period_min", "period_max",
               "theta_gain", "theta_clip")
_KNOT_KEYS = ("period_knots", "theta_knots")


def build_cfg(raw) -> dict:
    """conf の `cognition.age` ブロックを型強制つきで正準化する(既定 OFF)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
        elif k in _KNOT_KEYS:
            pts = [[float(x), float(y)] for x, y in (v or [])]
            cfg[k] = sorted(pts, key=lambda p: p[0])
    return cfg


def cfg_of(sim) -> dict:
    """設定を返す(初回のみ `sim.cfg` から遅延構築してキャッシュ。status.cfg_of と同型)。"""
    cfg = getattr(sim, "_agecogcfg", None)
    if cfg is None:
        raw = (sim.cfg.get("cognition", {}) or {}).get("age", {})
        cfg = build_cfg(raw)
        sim._agecogcfg = cfg
    return cfg


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


def profile_for(agent, cfg: dict) -> dict:
    """agent の年齢から係数を precompute して agent へキャッシュ(**乱数を引かない**)。

    media.profile_for / economy の precompute と同じ設計(人口統計属性のみ = R9)。"""
    prof = getattr(agent, "_age_cog", None)
    if prof is not None:
        return prof
    prof = _reg.age_cognition_params(int(getattr(agent, "age", 38) or 38), cfg)
    agent._age_cog = prof
    return prof


def period_mult(sim, agent) -> float:
    """定期発火の基本周期にかける個体倍率。**OFF は厳密に 1.0**(= 掛け算が恒等)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return 1.0
    return float(profile_for(agent, cfg)["period_mult"])


def threshold_delta(sim, agent) -> float:
    """実効閾値へ足す不透明な delta。**OFF は厳密に 0.0**(= 加算が恒等)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return 0.0
    return float(profile_for(agent, cfg)["theta_delta"])
