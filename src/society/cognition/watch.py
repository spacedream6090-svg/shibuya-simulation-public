"""監視仕様 watch spec(第82バッチ 2026-08-01・**既定 OFF**)= 期待値 ô の出どころ。

正典
----
- docs/plans/source/cognition-design-record.md **§2.2**(監視仕様)・§2.3(S 判定)・
  §2.8(責務分界: 期待値 ô は **LLM**、感度 g は **コード**)・付録(却下案)
- docs/plans/cognition-physics-plan.md **§6-3**(ユーザー修正: 予測誤差が大きいときに
  起きるのは「単に考える」ではなく**世界モデルの書き換え**= model-revision)

何を解く問題か
--------------
第81バッチの ô は **persistence(前回の認知イベント時点の観測値)** というプレースホルダ
だった。設計 §2.2 は違うことを言っている:

  1. **期待値 `ô_ic`** — 観測チャンネル c ごとの「こうなるはず」。**LLM が出力する。**
  2. **名前付きトリガ** — 「X が起きたら起こせ」の宣言的条件。**制約付き DSL。自由文は不可。**
  3. 感度 `g_ic` は監視仕様に**含めない**(LLM は出力しない = §2.4/§2.8)。
  4. **有効期限は持たせない**(時間で失効させると時間軸が裏口から入る)。上書きは次の発火時のみ。
  5. 「DSL は必ずホワイトリスト + 数値クランプで検証し、**不正出力なら前回仕様を維持**」。

本 module はこの 5 点だけを担う。S の計算・発火判定は `fire.py`、g の更新は
`plasticity.py`。

構造化出力の受け方(実装前リサーチの帰結)
------------------------------------------
LLM に機械可読な宣言を出させる標準は **constrained decoding / JSON Schema**(制約付き復号=
不正トークンをマスクして文法に合う出力だけを生成させる)であり、フレームワークは
Guidance / Outlines / llama.cpp / XGrammar / OpenAI / Gemini がこの形に収束している
(Geng et al. 2025, *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for
Language Models*, arXiv:2501.10868 — 6 実装の比較で「JSON Schema が事実上の共通言語」)。
一方で **形式の強制は内容の質を下げうる**ことも報告されている(Tam et al. 2024,
*Let Me Speak Freely?*, arXiv:2408.02442 — 過度に厳しい形式制約が推論課題の正答率を下げる)。

本リポジトリの立場: **復号器を差し替えない**(vLLM/Ollama/Anthropic を跨ぐため、また
「行動の自由記述」こそが研究対象なので出力全体に文法を被せたくない)。代わりに

  * プロンプトに **狭い形と選べる記号の一覧**だけを示し(自由文は受けない)、
  * 受け側で **ホワイトリスト検証 → 数値クランプ**を必ず通し、
  * 落ちたら **前回仕様を維持**する(捏造の既定値を作らない)

という「寛容に生成させ、厳格に受ける」構成を採る。これは制約付き復号の代替として
標準的な後段検証(schema validation + repair)の形であり、既存の `parse_action`
(壊れた JSON は routine へ後退)と同じ流儀でもある。

★ **チャンネル id はプロンプトに出さない**(no-fingerprint の要)
------------------------------------------------------------------
`body.state.*` のチャンネル id は factors 層のキーから実行時に生成される = **因子名を
含む**。これをそのままプロンプトに書くと、認知層が因子を名指ししない契約
(tests/test_contracts.py)を実質的に破る。したがって watch spec では

    c01, c02, … = 不透明な記号(usable チャンネルの並び順で採番)

だけをプロンプトに出し、人間可読な意味は `Channel.label`(因子名を含まない中立ラベル)で
与える。対応表は run_manifest.json に載せるので事後解析はできる。

R1 ドクトリン
-------------
- 既定 `cognition.watch.enabled: false` では本 module は **1 度も呼ばれない**
  (`enabled()` が False → プロンプトに 1 行も足さず、応答も読まない)= バイト一致。
- 乱数を **1 本も引かない**(本 module は RngHub に触れない)。
- LLM 呼び出しを**増やさない**(既存の発火 1 回の出力に節が 1 つ増えるだけ)。
- ON は **journal 等級・affects_k=true**: ô が変われば S が変わり、驚き発火の本数が変わる。
- ON は **fingerprint_risk=possible** を正直に宣言する: watch 節そのものが全発火プロンプトに
  出る(= 条件の有無に気づく余地が原理的にある)し、model-revision の 1 行は**驚き発火の
  ときだけ**出る(= その思考が予測外れに起因することを本人が知りうる)。これは §6-3 が
  要求した設計そのもの(別種の認知モード)なので、隠すのではなく宣言する。
"""
from __future__ import annotations

import math

from ..observer.schema import Event
from . import channels as _channels

SCHEMA = 1

# プロンプトに載せる JSON キー(mock backend の分岐マーカーも兼ねる)。
KEY = "watch"
MARK = '"watch"'

# 受理する比較演算子(**ホワイトリスト**。自由文・式・関数呼び出しは受けない)。
OPS: tuple[str, ...] = (">", ">=", "<", "<=")
_OP_FN = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}
# 別名(実 LLM は "gt" / "＞" などを返しうる。**綴りの揺れだけ**を吸収し、意味は増やさない)。
_OP_ALIAS = {"gt": ">", "ge": ">=", "gte": ">=", "lt": "<", "le": "<=", "lte": "<=",
             "＞": ">", "＜": "<", "≧": ">=", "≦": "<=", ">=": ">=", "<=": "<="}

# 単位ごとの許容域(**数値クランプの外枠**)。ここを外れた値は「不正出力」として扱う。
_DOMAIN: dict[str, tuple[float, float]] = {
    "count": (0.0, 10_000.0),
    "level": (0.0, 1.0),
    "flag": (0.0, 1.0),
    "degC": (-50.0, 60.0),
}

# 監視仕様の状態("なし"と"読めなかった"を区別する)。
OK, ABSENT = "ok", "absent"
REJECT_SHAPE, REJECT_CHANNEL = "reject_shape", "reject_channel"
REJECT_OP, REJECT_VALUE = "reject_op", "reject_value"
STATUSES = (OK, ABSENT, REJECT_SHAPE, REJECT_CHANNEL, REJECT_OP, REJECT_VALUE)


# --------------------------------------------------------------------------- #
# cfg 正準化(既定 OFF)
# --------------------------------------------------------------------------- #
def build_cfg(raw: dict | None) -> dict:
    """conf の `cognition.watch` ブロックを型強制つきで正準化する(既定 OFF)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        # 期待値の影響を σ 何本ぶんまでに抑えるか(**数値クランプ**。設計 §2.2)。
        # ô が観測から離れすぎると 1 チャンネルが S を独占して閾値方式が壊れる。
        "clamp_sigmas": max(0.0, float(raw.get("clamp_sigmas", 6.0))),
        # 1 個体が同時に持てる名前付きトリガの本数(超過分は先頭から採る)。
        "max_triggers": max(0, int(raw.get("max_triggers", 3) or 0)),
        # トリガ 1 本の重み w_ij(単位は σ。**LLM は設定しない**= §2.4 と同じ理由)。
        "trigger_weight": max(0.0, float(raw.get("trigger_weight", 1.0))),
        # トリガ名の最大長(自由文の書き込み先にしないための上限)。
        "name_max": max(0, int(raw.get("name_max", 24) or 0)),
        # model-revision(§6-3): 驚き発火のプロンプトに中立な 1 行を足す。
        "model_revision": bool(raw.get("model_revision", True)),
        # model-revision のとき、未検証の伝聞信念の確信度に掛ける係数(1.0=何もしない)。
        # 実際に信念へ触れるのは beliefs.enabled ON のときだけ(engine 側が呼ぶ)。
        "belief_revision": max(0.0, min(1.0, float(raw.get("belief_revision", 0.9)))),
        "belief_max_facts": max(0, int(raw.get("belief_max_facts", 5) or 0)),
    }


def enabled(sim) -> bool:
    """watch spec が有効か。**発火機構(fire)が ON であることが前提**。"""
    cfg = getattr(sim, "watchcfg", None)
    if not (cfg and cfg["enabled"]):
        return False
    from . import fire as _fire
    return _fire.enabled(sim)


def model_revision_on(sim) -> bool:
    return bool(enabled(sim) and sim.watchcfg["model_revision"])


# --------------------------------------------------------------------------- #
# 不透明な記号 ↔ チャンネル(プロンプトに id を出さないための対応表)
# --------------------------------------------------------------------------- #
def symbols(sim) -> tuple[tuple[str, int, str, float, str], ...]:
    """`(記号, 観測タプル内の位置, channel_id, σ_c, ラベル)` の列(run 内で不変)。

    並びは `fire.usable_channels` と同一(= σ_c 凍結ファイルの usable を id 昇順)。
    記号は位置だけで決まるので、同じ σ_c ファイルなら run 間でも安定する。
    """
    cached = getattr(sim, "_watch_symbols", None)
    if cached is not None:
        return cached
    from . import fire as _fire
    out = tuple(
        (f"c{n + 1:02d}", i, cid, sigma,
         (_channels.BY_ID[cid].label if cid in _channels.BY_ID else cid))
        for n, (i, cid, sigma) in enumerate(_fire.usable_channels(sim)))
    sim._watch_symbols = out
    return out


def _by_symbol(sim) -> dict:
    cached = getattr(sim, "_watch_by_symbol", None)
    if cached is None:
        cached = {sym: (i, cid, sigma) for sym, i, cid, sigma, _lab in symbols(sim)}
        sim._watch_by_symbol = cached
    return cached


# --------------------------------------------------------------------------- #
# プロンプト(**中立提示**: 勧めない・評価語を書かない・条件語彙を出さない)
# --------------------------------------------------------------------------- #
_INTRO = ("この後しばらくの見通しを、上の JSON に 1 つだけ足して書く"
          "(この形だけ。文章では書かない):")
_FORM = ('  "watch": {"expect": {"c01": 3, "c02": 0.4}, '
         '"triggers": [{"name": "短い名前", "ch": "c03", "op": ">", "value": 5}]}')
_RULE = ("  expect = それぞれが次にどれくらいの値になると思うか(数値だけ)。"
         "分からない項目は書かない。\n"
         "  triggers = 「こうなったら気づきたい」条件。op は > >= < <= のどれか。"
         "無ければ [] でよい。")
_ITEMS = "  書ける項目(この記号だけを使う):"

# model-revision(§6-3)。**誘導語彙なし**: 何をどう考え直すかは一切指定しない。
# 「予測」「期待値」「驚き」「発火」等の機構語も、実験条件を示す語も使わない。
_REVISION_LINE = "思っていたのと違うことが起きた。見立てを書き直してよい。"


def section(sim) -> str | None:
    """watch 節のプロンプト断片(OFF は None = 1 行も足さない)。

    run 内で不変なので 1 度だけ組んで使い回す(APC の prefix 一致を壊さない)。
    """
    if not enabled(sim):
        return None
    cached = getattr(sim, "_watch_section", None)
    if cached is not None:
        return cached
    syms = symbols(sim)
    if not syms:
        sim._watch_section = None
        return None
    lines = [_INTRO, _FORM, _RULE, _ITEMS]
    lines += [f"    {sym} = {label}" for sym, _i, _cid, _sigma, label in syms]
    text = "\n".join(lines)
    sim._watch_section = text
    return text


def revision_line(sim, agent, step: int, trigger: str) -> str | None:
    """model-revision の 1 行(該当しないときは None = 1 行も足さない)。

    条件: watch ON かつ model_revision ON かつ **この step の発火理由が驚きだった**こと。
    返答(reply)は相手起点の発話なので対象外(自分の予測が外れた話ではない)。
    """
    if not model_revision_on(sim) or trigger == "reply":
        return None
    from . import fire as _fire
    src = getattr(agent, "_fire_src", None)
    if src is None or int(src[0]) != int(step) or src[1] != _fire.SALIENCE:
        return None
    return _REVISION_LINE


# --------------------------------------------------------------------------- #
# 受理(ホワイトリスト検証 → 数値クランプ → 不正なら前回仕様を維持)
# --------------------------------------------------------------------------- #
def _number(value) -> float | None:
    """数値だけを受ける(bool は数値として受けない・非有限は不正)。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _clamp_expect(value: float, obs, idx: int, sigma: float,
                  span: float) -> tuple[float, bool]:
    """ô を「いまの観測から σ 何本ぶんまで」に抑える(数値クランプ)。

    観測が欠測(None)のチャンネルは基準が無いのでクランプしない(検証は通っている)。
    """
    if span <= 0.0 or obs is None:
        return value, False
    now = obs[idx] if idx < len(obs) else None
    if now is None:
        return value, False
    lo, hi = float(now) - span * sigma, float(now) + span * sigma
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False


def parse(sim, agent, response: str) -> tuple[dict | None, str, int]:
    """応答から watch 節を取り出す。返り値 `(spec, status, n_clamped)`。

    `spec` は `{"expect": {位置: ô}, "triggers": [(名前, 位置, op, 閾値)]}`。
    **spec が None のときは呼び手が前回仕様をそのまま残すこと**(設計 §2.2)。
    副作用ゼロ・乱数ゼロ・LLM ゼロの純関数。
    """
    from . import deliberate as _deliberate
    data = _deliberate._loads_lenient(response)
    if not isinstance(data, dict) or KEY not in data:
        return None, ABSENT, 0
    block = data.get(KEY)
    if not isinstance(block, dict):
        return None, REJECT_SHAPE, 0

    cfg = sim.watchcfg
    table = _by_symbol(sim)
    obs = getattr(agent, "_fire_obs", None)
    span = float(cfg["clamp_sigmas"])
    clamped = 0

    # ---- expect: {記号: 数値} ----
    raw_expect = block.get("expect", {})
    if raw_expect is None:
        raw_expect = {}
    if not isinstance(raw_expect, dict):
        return None, REJECT_SHAPE, 0
    expect: dict[int, float] = {}
    for sym in sorted(str(k) for k in raw_expect):          # 走査順を出力順に依存させない
        cell = table.get(sym)
        if cell is None:
            return None, REJECT_CHANNEL, 0                  # ホワイトリスト外
        value = _number(raw_expect[sym])
        if value is None:
            return None, REJECT_VALUE, 0
        idx, cid, sigma = cell
        lo, hi = _DOMAIN.get(_channels.BY_ID[cid].unit, (-1e9, 1e9))
        if not (lo <= value <= hi):
            return None, REJECT_VALUE, 0                    # 単位の許容域外 = 不正出力
        value, hit = _clamp_expect(value, obs, idx, sigma, span)
        clamped += int(hit)
        expect[idx] = value

    # ---- triggers: [{name, ch, op, value}] ----
    raw_trigs = block.get("triggers", [])
    if raw_trigs is None:
        raw_trigs = []
    if not isinstance(raw_trigs, (list, tuple)):
        return None, REJECT_SHAPE, 0
    triggers: list[tuple[str, int, str, float]] = []
    for item in list(raw_trigs)[:max(0, int(cfg["max_triggers"]))]:
        if not isinstance(item, dict):
            return None, REJECT_SHAPE, 0
        cell = table.get(str(item.get("ch", item.get("channel", ""))))
        if cell is None:
            return None, REJECT_CHANNEL, 0
        op = str(item.get("op", "")).strip()
        op = _OP_ALIAS.get(op.lower(), op)
        if op not in OPS:
            return None, REJECT_OP, 0                       # 自由文・未知演算子は不可
        value = _number(item.get("value"))
        if value is None:
            return None, REJECT_VALUE, 0
        idx, cid, sigma = cell
        lo, hi = _DOMAIN.get(_channels.BY_ID[cid].unit, (-1e9, 1e9))
        if not (lo <= value <= hi):
            return None, REJECT_VALUE, 0
        name = str(item.get("name", "") or "").strip()[:int(cfg["name_max"])]
        triggers.append((name, idx, op, value))

    if not expect and not triggers:
        return None, ABSENT, 0                              # 空の節は「無かった」と同じ
    return {"expect": expect, "triggers": triggers}, OK, clamped


def apply(sim, agent, response: str, step: int, sim_min: int) -> str:
    """応答を検証して監視仕様を差し替える(不正なら**前回仕様を維持**)。

    L1 `watch_spec` を 1 件残す(ON のときだけ生える kind なので OFF は無風)。
    「読めなかった」ことそのものが観測量(構造化出力の遵守率)なので握り潰さない。
    """
    if not enabled(sim):
        return ABSENT
    spec, status, clamped = parse(sim, agent, response)
    if spec is not None:
        agent._fire_watch = spec
    payload = {"status": status, "clamped": int(clamped)}
    if spec is not None:
        payload["n_expect"] = len(spec["expect"])
        payload["n_trigger"] = len(spec["triggers"])
        if spec["triggers"]:
            payload["names"] = [nm for nm, _i, _o, _v in spec["triggers"]]
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                         kind="watch_spec", x=agent.x, y=agent.y, payload=payload))
    return status


# --------------------------------------------------------------------------- #
# 消費(fire.py が S を組むときに呼ぶ)
# --------------------------------------------------------------------------- #
def expectation(sim, agent, base_pred):
    """ô ベクトル = 「LLM が言った期待値」を persistence 予測の上に重ねたもの。

    LLM が触れなかったチャンネルは persistence(第81 の既定)のまま残す。
    watch 仕様が無い個体は `base_pred` をそのまま返す(= 第81 と同一挙動)。
    """
    spec = getattr(agent, "_fire_watch", None)
    if not spec or not spec.get("expect"):
        return base_pred
    if base_pred is None:
        # まだ 1 度も観測を凍結していない個体。ô だけで S を組むと欠測チャンネルが
        # 全部落ちるので、長さだけ合わせた None 埋めの土台を作る。
        pred = [None] * len(_channels.CHANNELS)
    else:
        pred = list(base_pred)
    for idx, value in spec["expect"].items():
        if 0 <= idx < len(pred):
            pred[idx] = value
    return tuple(pred)


def trigger_term(sim, agent, obs) -> tuple[float, list[str]]:
    """Σ_j w_ij·[trigger_j 成立] と、成立したトリガ名の一覧。

    重み w_ij は **conf の定数**(LLM は設定しない)。欠測チャンネルは成立しない。
    """
    spec = getattr(agent, "_fire_watch", None)
    if not spec or not spec.get("triggers") or obs is None:
        return 0.0, []
    weight = float(sim.watchcfg["trigger_weight"])
    total, names = 0.0, []
    for name, idx, op, value in spec["triggers"]:
        if not (0 <= idx < len(obs)):
            continue
        now = obs[idx]
        if now is None:
            continue
        if _OP_FN[op](float(now), float(value)):
            total += weight
            names.append(name)
    return total, names


# --------------------------------------------------------------------------- #
# 来歴(run_manifest.json 用)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """監視仕様の宣言。OFF(既定)では None = 既存 manifest と同形。"""
    if not enabled(sim):
        return None
    cfg = sim.watchcfg
    return {
        "schema": SCHEMA,
        "expectation_model": "llm_watch_spec(未指定チャンネルは persistence へ後退)",
        "clamp_sigmas": cfg["clamp_sigmas"],
        "ops": list(OPS),
        "max_triggers": cfg["max_triggers"],
        "trigger_weight": cfg["trigger_weight"],
        "model_revision": bool(cfg["model_revision"]),
        "belief_revision": cfg["belief_revision"],
        # ★ プロンプトに出す不透明記号 ↔ チャンネルの対応表(事後解析用。id は出さない
        #   という契約はプロンプト側の話で、来歴には正直に残す)
        "symbols": {sym: cid for sym, _i, cid, _s, _lab in symbols(sim)},
        "prompt_sha256": _section_sha(sim),
    }


def _section_sha(sim) -> str:
    import hashlib
    text = section(sim) or ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

