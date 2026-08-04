"""痕跡 = 場所のイベント履歴 IF-D(``world.traces``・**既定 OFF**)。

正典
----
- docs/research/if-lane-research.md **§1**(スティグマジー。Grassé 1959 → Heylighen 2011/2016 →
  Parunak のデジタルフェロモン。★**演算は集約と蒸発の 2 つだけ**にする = Parunak の
  **propagation factor = 0** として文献的に正当。★減衰は欠陥ではなく**機能**:
  「枯れた餌場を指すフェロモン道は『無関係になった』のではなく、無駄な旅を誘発する点で
  **誤誘導(misleading)**になった」。減衰速度は「**情報が陳腐化する速度**」に合わせる
  = 単一の TTL 既定値を全種に使うのは文献に反する → **TTL 3 階層**。
  ★既存の貼り紙(flyer)は node × TTL × 容量 × narrowcast の**完全なスティグマジー実装**で、
  本 module はその構造を「意図的な掲示」から「行為の副産物」へ一般化したものである)
- docs/research/llm-world-interface-audit.md **§3**(痕跡カテゴリの現状 = 貼り紙 / 出店 /
  場所ラベルの個別 3 種のみ。「場所のイベント履歴」は **L1 に座標付きで全部残るが
  シム内から読む経路がゼロ** = Heylighen の用語では「medium ではない」状態そのもの
  = 変えられるが知覚できない)
- docs/plans/if-sv-p4-plan.md 「IF-D」行

何を解く問題か
--------------
街で起きたことは L1 に残るが、**後から来た者はそれを知る手段を持たない**。摘発があった
路地も、昨日まで人が集まっていた広場も、シムの中では何の跡も残らない。Heylighen の
定義に照らすと、本シムの node は「変更できるが知覚できない」= medium ではなかった。
IF-D は node を medium にする。行為が node に**強度**を残し(集約)、時間が経てば
薄れ(蒸発)、その場に居合わせなかった者が 1 行として知覚する。

★**propagation(拡散)はコードとして存在させない。**
  Parunak の枠組みでは place agent の場に対する演算は aggregation / propagation /
  evaporation の 3 つだが、**propagation factor = 0** は正当な設定である。
  本シムで 0 を採る理由は if-lane-research §1-3 の 3 点:
    (a) node グラフ上の拡散は近傍列挙順に依存し golden のバイト一致を壊しやすい、
    (b) 25 万体で O(node × kind) の毎 step 掃引が乗る、
    (c) 「隣の駅に噂が滲む」は**情報オブジェクト(IF-C = rumors.py)の担当**であって
        痕跡の担当ではない。
  したがって本 module には近傍・隣接・拡散に相当する識別子が 1 つも無い
  (tests/test_traces.py が AST で機械固定する)。

役割分担(既存の 4 機構と混ぜない)
------------------------------------
| 機構 | 何に紐づくか | 誰に届くか | 生まれ方 | 消え方 |
|---|---|---|---|---|
| **traces(本 module)** | **場所**(node) | その node に**後から**来た者 | 行為の**副産物** | 蒸発(半減期・3 階層) |
| flyer(tools.py) | 場所(node) | その node に来た者 | **意図的な掲示**(post_flyer) | TTL 144 step + 容量押し出し |
| rumors(rumors.py) | **出来事**(Item) | 人 → 人(会話に相乗り) | 構造化イベント | stifler 化 + 忘却 TTL |
| gossip(gossip.py) | **人**(匿名タグ) | 知人 → 知人(complex contagion) | 負イベントの当事者 | 忘却 |
| place_bind(labels) | 場所(node) | その node に来た者 | 造語の**束縛** | 消えない(語の定着) |

要するに traces は「**誰のものでもない場所の記憶**」で、flyer(誰かが貼った)・
rumors(誰かが語った)・gossip(誰かの評判)のいずれとも源が違う。

3 つの設計判断(と、その理由)
------------------------------
**(1) 源は構造化イベントに限る。自由文は 1 バイトも読まない。**
    痕跡の文面は **(痕跡種) だけの純関数**で、源イベントの自由文
    (``event_host`` の title・``venture_open`` の name・``nuisance`` の kind)は
    1 バイトも使わない。rumors.py の設計判断 (1) と同じ理由で、自由文を通せば
    (i) 等級が strict から journal へ落ち、(ii) 数字・記号が文面に混ざり、
    (iii) 自由文の世界書き込み帯域が 1 本増える。

**(2) TTL は 3 階層(Heylighen の「情報が陳腐化する速度」)。**
    transient(半減期 ~3 時間 = 揉め事・事故級)/ daily(144 step = 1 日 = 出来事・
    摘発級 = 既存 flyer と同水準)/ persistent(半減期 0 = **減衰しない** = 場所に
    定着した性格)。種 → 階層の割当は conf(``tier_of``)で差し替えられる。
    ★蒸発は**日境界に 1 回だけ**適用する(第75 dunbar が「接触時適用は振動 5739 件を
      実測して棄却」した前例に合わせる)。経過 step から半減期で決定論的に減衰させるので、
      日境界がずれても(resume でも)同じ強度になる。

**(3) 当事者(発生時に同席していた者)には痕跡の行を出さない。**
    自分が見た出来事を「最近あったようだ」と伝聞調で二重に読まされるのは不自然なので、
    集約の時点で同席者(``hearers_of`` = 既存の知覚半径規約)を痕跡レコードに控え、
    その者には行を出さない。★**控えるのは集約(=行為の効果)の時点だけ**で、
    観測(=行を読むこと)では痕跡を 1 ミリも動かさない(if-lane-research §1-3 の示唆 6:
    「痕跡の強化は行為の副産物としてのみ起こし、観測では絶対に起こさない」。
    Heylighen の定義でも trace は action の効果であって perception の効果ではない)。

**(4) ``ablate.propagation_off`` では遮断しない(rumors とは扱いが違う)。**
    同 ablation の契約は「発話は生成するが**その内容**が他エージェントの文脈に入らない」
    (= **送り手の自由文**の遮断)である。``rumors.on_talk`` と ``tools._view_flyers`` が
    遮断されるのは、どちらも**誰かが書いた/語った内容**を聞き手の記憶へ運ぶからで、
    痕跡の 1 行は誰の文でもない(``sentence(kind)`` = 有限テンプレの純関数で、
    自由文を 1 バイトも運ばない)。同じ理由で ``crowd_line`` / ``place_label_line`` /
    出店の存在も遮断されていない。**世界がそうなっている事実**は propagation_off の
    対象外である、という既存の線引きに揃える。

R1 ドクトリン
-------------
- 既定 ``enabled: false`` では ``phase`` / ``line_for`` が即 return し、
  **L1 に 1 件も出ず・記憶に 1 行も入らず・agent に属性が生えず・sim に state が生えず・
  プロンプトが 1 バイトも変わらない**(= ゴールデン L1 バイト一致。第75 dunbar の dormant /
  第94 IF-B / 第95 IF-C と同じ「OFF ではキー自体を作らない」流儀)。
- **乱数 stream を 1 本も引かない**(集約も蒸発も完全決定論)。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。
- no-fingerprint: 痕跡の 1 行は ``sentence(kind)`` = **痕跡種だけの純関数**で、
  config・実験条件・k・特性・時刻・**強度の数値**を 1 つも読まない。同じ種からは
  全実験条件で同一バイト列になる(tests/test_traces.py が機械固定)。
  ★強度・件数・階層名は**絶対にプロンプトへ入れない**(観測側だけの概念に閉じる)。

正直な限界(4 件)
------------------
- **1 step の遅れ**: 痕跡の集約は step 末の L1 走査で起きるので、その step の思考には
  乗らない(最短で次 step から読める)。「後から来た者」の機構なので実害は小さいが、
  同 step 内に居合わせた者と痕跡で知る者の境目は 1 step ぶんぼやける。
- **閲覧イベントを出さない**: 痕跡を読んだことは L1 に残さない(``flyer_view`` と違い
  受動観測で、毎 step 全在場者ぶん発生して L1 を埋め尽くすため)。読まれた**総数**だけを
  ``summary.traces.lines_injected`` に残す。
- **当事者の控えは live な痕跡レコードにだけ残る**: 蒸発しきった痕跡は同席者名簿ごと
  消えるので、persistent 階層の痕跡だけが同席者名簿を長く保持する(件数は「その node で
  出店/閉店が起きた回数 × その場の人数」で、実測上は小さい)。
- **``max_per_step`` は安全弁**: 1 step に集約する件数に上限を置く(暴走時の L1 膨張対策)。
  超過分は**捨てる**(次 step へ繰り越さない)= 決定論を単純に保つため。

**絶対に触らない範囲(明文規約)**
--------------------------------
1. ``observer/metrics_spec.py`` の**凍結 14 ファイル**は 1 バイトも変更しない。
2. ``tools.py`` の flyer 実装は 1 バイトも変更しない(本 module は flyer の**一般化**だが、
   既存実装を書き換えるのではなく**並置**する。flyer は「意図的な掲示」で痕跡は
   「行為の副産物」= Heylighen の marker / trace の区別に対応しており、統合すると
   その区別が消える)。
"""
from __future__ import annotations

from .observer.schema import Event
from .world.perception import hearers_of

SCHEMA = 1

# ---- TTL 3 階層(Heylighen: 減衰速度は情報が陳腐化する速度に合わせる)------------- #
TRANSIENT = "transient"      # 半減期 ~3 時間: 揉め事・事故級(すぐ陳腐化する)
DAILY = "daily"              # 半減期 1 日  : 出来事・摘発級(既存 flyer と同水準)
PERSISTENT = "persistent"    # 半減期 ∞     : 場所に定着した性格(商いの有無など)
TIERS: tuple[str, ...] = (TRANSIENT, DAILY, PERSISTENT)


# --------------------------------------------------------------------------- #
# 痕跡種の有限表(**ここに無い L1 種からは痕跡が生まれない**)
#
#   sources … 源にする L1 イベント種(複数可)
#   tier    … 既定の TTL 階層(conf の ``tier_of`` で差し替え可)
#   word    … 定型語。**出来事の種類だけ**を指す語で、数値・固有名・自由文は入らない
#
# ★Parunak の flavor(意味の異なる場を混ぜない)に従い、種ごとに独立の強度を持つ。
#   「賑わい」と「揉め事」を 1 つのスカラーに足し込まない。
# --------------------------------------------------------------------------- #
TRACE_SPECS: dict[str, tuple[tuple[str, ...], str, str]] = {
    # 公共の出来事(その場に居れば誰の目にも触れる)
    "enforcement": (("enforcement",),   DAILY,      "取り締まり"),
    "gathering":   (("event_host",),    DAILY,      "人の集まり"),
    "opening":     (("venture_open",),  PERSISTENT, "新しい店の開業"),
    "closing":     (("venture_close",), PERSISTENT, "店じまい"),
    "trouble":     (("crime", "nuisance"), TRANSIENT, "揉め事"),
}

# 既定の痕跡種(**表の全種 = すべて公共の出来事**)。
#
# なぜ ``chance_event`` を源にしないか:
#   if-sv-p4-plan の IF-D 行は「chance_event(事件系があれば)」と条件つきで挙げているが、
#   実装の ``chance_event`` は windfall(臨時収入)/ loss(財布紛失)/ encounter(偶然の
#   出会い)の 3 種で、**当人にしか起きない私的な出来事**であり payload に node も無い。
#   「その場所で何かがあった」の痕跡にすると、通りすがりの人が他人の落とし物を
#   場所の性格として読むことになる。街の**事件系**として妥当な L1 種は ``crime``
#   (窃盗)と ``nuisance``(客引き・ナンパ・喧嘩・騒ぎ)なので、そちらを trouble に
#   割り当てた(どちらも既定 OFF の diversity 層が出す = 二重に保守的)。
DEFAULT_KINDS: tuple[str, ...] = tuple(TRACE_SPECS)

# 源 L1 イベント種 → 痕跡種(逆引き。**多対一**を許す = trouble は 2 種を束ねる)
SRC_TO_KIND: dict[str, str] = {
    src: kind for kind, (srcs, _t, _w) in TRACE_SPECS.items() for src in srcs}

# 当事者として追加で控える payload キー(**同席者に加える**。id の列)。
# ★源イベントの自由文キー(title / name / kind)は 1 つも読まない。
ACTOR_KEYS: tuple[str, ...] = ("target", "officer", "victim", "offender", "owner")

# 定型文(**痕跡種だけの純関数**。強度・件数・階層名・時刻・固有名は 1 つも入らない)
TEXT = "この場所では最近、{word}があったようだ。"

DEFAULTS = {
    "enabled": False,
    "kinds": list(DEFAULT_KINDS),
    "tier_of": {},              # 種 → 階層の上書き(空 = TRACE_SPECS の既定)
    "half_life_steps": {TRANSIENT: 18, DAILY: 144, PERSISTENT: 0},
    "deposit": 1.0,             # 1 件の出来事が置く強度
    "max_strength": 3.0,        # 同種の上限(クリップ)
    "line_threshold": 1.0,      # これ以上の強度の痕跡だけが 1 行になる
    "drop_threshold": 0.05,     # これ未満に薄れた痕跡は消える(明示的失効ロジック)
    "max_per_step": 32,         # 1 step に集約する件数の上限(安全弁)
}

_FLOAT_KEYS = ("deposit", "max_strength", "line_threshold", "drop_threshold")


# --------------------------------------------------------------------------- #
# cfg 正準化(rumors.build_cfg / gossip.build_cfg と同型: dict / OmegaConf 両対応)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の ``world.traces`` ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(``rumors.build_cfg`` と同じ作法)。
    ``kinds`` は**置き換え**(重ねない): 何を痕跡にしたランなのかが config から一意に
    読めるようにするため。``TRACE_SPECS`` に無い種は**黙って捨てる**(定型語を持たない
    種から痕跡は作れない = 捏造しない)。``tier_of`` の未知の階層名も捨てる。
    """
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    cfg["kinds"] = list(DEFAULT_KINDS)
    cfg["tier_of"] = {}
    cfg["half_life_steps"] = dict(DEFAULTS["half_life_steps"])
    for k, v in raw.items():
        if k == "enabled":
            cfg["enabled"] = bool(v)
        elif k == "kinds":
            cfg["kinds"] = [str(s) for s in (_to_plain(v) or [])
                            if str(s) in TRACE_SPECS]
        elif k == "tier_of":
            cfg["tier_of"] = {str(a): str(b)
                              for a, b in dict(_to_plain(v) or {}).items()
                              if str(a) in TRACE_SPECS and str(b) in TIERS}
        elif k == "half_life_steps":
            got = dict(_to_plain(v) or {})
            for tier in TIERS:
                if tier in got:
                    cfg["half_life_steps"][tier] = max(0, int(got[tier]))
        elif k in _FLOAT_KEYS:
            cfg[k] = max(0.0, float(v))
        elif k == "max_per_step":
            cfg["max_per_step"] = max(0, int(v))
    cfg["max_strength"] = max(cfg["max_strength"], cfg["deposit"])
    return cfg


def cfg_of(sim) -> dict:
    """痕跡設定(初回のみ ``sim.cfg.world.traces`` から遅延構築してキャッシュ)。

    simulation.py は読み取り専用なので cfg は本 module が遅延構築する
    (``rumors.cfg_of`` / ``gossip.cfg_of`` と同型)。キャッシュ属性 ``sim.tracescfg`` は
    L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。
    """
    c = getattr(sim, "tracescfg", None)
    if c is None:
        try:
            raw = (sim.cfg.get("world", None) or {}).get("traces", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        c = build_cfg(raw)
        sim.tracescfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


def tier_of(cfg: dict, kind: str) -> str:
    """痕跡種の TTL 階層(conf の上書き > TRACE_SPECS の既定)。"""
    over = cfg.get("tier_of") or {}
    got = over.get(str(kind))
    if got in TIERS:
        return str(got)
    spec = TRACE_SPECS.get(str(kind))
    return spec[1] if spec is not None else DAILY


# --------------------------------------------------------------------------- #
# 定型文(純関数。sim も agent も config も強度も見ない = 全実験条件で同一バイト列)
# --------------------------------------------------------------------------- #
def sentence(kind: str) -> str:
    """痕跡の 1 行。**痕跡種だけの純関数**。

    埋まるのは有限表の定型語だけで、強度・件数・階層名・時刻・金額・実験条件・機構語は
    1 つも入らない(tests が「テンプレート側にも定型語側にも数字が無い」ことを機械検査する)。
    """
    spec = TRACE_SPECS.get(str(kind))
    if spec is None:                               # 語彙外は行にしない(捏造しない)
        return ""
    return TEXT.format(word=spec[2])


# --------------------------------------------------------------------------- #
# 痕跡ストア(ON 経路でのみ生やす。checkpoint.py が runtime で中央管理する)
#
#   nodes[node][kind] = {"s": 強度, "step": 最終集約 step, "w": {当事者id: step}}
#
# ★挿入順は「node 昇順 → kind 昇順」に保つ(反復は常に sorted で行うので順序に依存しない
#   が、pickle 越しの差分を読みやすくするため)。
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_trace_state", None)
    if st is None:
        st = {"schema": SCHEMA, "day": -1, "evap_step": -1,
              "marks": 0, "evaporated": 0, "lines": 0,
              "by_kind": {}, "lines_by_kind": {}, "nodes": {}}
        sim._trace_state = st
    return st


def _bump(table: dict, key: str) -> None:
    table[key] = int(table.get(key, 0)) + 1


# --------------------------------------------------------------------------- #
# 演算 1/2: 集約(aggregation)= 対象イベント発生時の強度加算 + 同種の上限クリップ
# --------------------------------------------------------------------------- #
def _new_events(sim) -> list:
    """前回処理済み総数(watermark)以降の新規 L1(``rumors._new_events`` と同型)。

    watermark は**プロセス内 logger カウンタ由来**なので checkpoint に保存しない
    (resume 直後は新しい logger の 0 から数え直す = 第59 assets / 第61 gossip と同流儀)。
    """
    logger = getattr(sim, "logger", None)
    if logger is None:
        return []
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)
    processed = int(getattr(sim, "_trace_watermark", 0))
    new = max(0, min(total - processed, len(events)))
    sim._trace_watermark = total
    return events[len(events) - new:] if new else []


def _bystanders(sim, actor, payload: dict) -> list[int]:
    """当事者 = 行為者 + payload の相手 + 同席者(**agent_id 昇順**の決定論・乱数ゼロ)。

    同席者 = 既存の「同席者」規約(``hearers_of`` = 知覚半径・睡眠中と屋内外の別を尊重)。
    ★ここで読むのは **id の欄だけ**で、payload の自由文(title / name / kind)は
      1 つも読まない。
    """
    ids: set[int] = set()
    if actor is not None:
        ids.add(int(actor.id))
    for key in ACTOR_KEYS:
        oid = payload.get(key)
        if isinstance(oid, int) and not isinstance(oid, bool) and oid >= 0:
            ids.add(int(oid))
    if actor is not None:
        radius = float(sim.cfg.world.perception_radius_m)
        index = getattr(sim, "percept_index", None)
        for w in hearers_of(actor, index if index is not None else sim.agents,
                            radius):
            ids.add(int(w.id))
    return sorted(ids)


def _aggregate(sim, cfg: dict, st: dict, e, step: int, sim_min: int) -> bool:
    """L1 イベント 1 件 → その node の痕跡へ強度を加算(集約しなければ False)。

    ★演算はここ(集約)と ``_evaporate``(蒸発)の **2 つだけ**。
      近傍への拡散(propagation)は**存在しない**(module docstring / Parunak factor 0)。
    """
    kind = SRC_TO_KIND.get(e.kind)
    if kind is None or kind not in cfg["kinds"]:
        return False
    payload = e.payload or {}
    actor = sim.agent_by_id.get(int(e.agent_id)) if int(e.agent_id) >= 0 else None
    node = payload.get("node") or (getattr(actor, "node", None) if actor else None)
    if not node:                                   # 場所が決まらない出来事は痕跡にしない
        return False
    node = str(node)
    per = st["nodes"].setdefault(node, {})
    rec = per.get(kind)
    if rec is None:
        rec = {"s": 0.0, "step": int(step), "w": {}}
        per[kind] = rec
    # ---- 集約: 同種の強度を足し、上限でクリップする(Parunak の aggregation)-------- #
    rec["s"] = min(float(cfg["max_strength"]),
                   float(rec["s"]) + float(cfg["deposit"]))
    rec["step"] = int(step)
    for aid in _bystanders(sim, actor, payload):   # 当事者を控える(観測では触らない)
        rec["w"].setdefault(int(aid), int(step))
    st["marks"] += 1
    _bump(st["by_kind"], kind)
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min),
                         agent_id=int(e.agent_id), kind="trace_mark",
                         x=e.x, y=e.y,
                         payload={"node": node, "kind": kind,
                                  "tier": tier_of(cfg, kind)}))
    return True


# --------------------------------------------------------------------------- #
# 演算 2/2: 蒸発(evaporation)= 半減期ベースの決定論減衰(**日境界 1 回**)
# --------------------------------------------------------------------------- #
def _evaporate(sim, cfg: dict, st: dict, step: int, sim_min: int) -> None:
    """日境界に 1 回だけ、経過 step ぶんの半減期減衰を掛ける(乱数ゼロ・完全決定論)。

    ★接触時・毎 step ではなく**日境界 1 回**にするのは第75 dunbar の前例
      (接触時適用は振動 5739 件を実測して棄却)に合わせるため。減衰量は
      「前回の蒸発からの経過 step」から計算するので、日境界がいつ来ても
      (= resume を跨いでも)同じ強度に落ち着く。
    ★``half_life_steps[tier] == 0`` は **persistent = 減衰しない**の意味
      (Ledger-State Stigmergy が言う「蒸発しない媒体」を**明示的に選んだ**場合)。
    ★drop_threshold 未満になった痕跡はレコードごと削除する(明示的失効ロジックの義務化。
      同席者名簿もここで一緒に消えるのでメモリが伸び続けない)。
    """
    day = int(sim_min) // 1440
    if day == int(st["day"]):
        return
    first = int(st["day"]) < 0
    st["day"] = day
    prev = int(st["evap_step"])
    st["evap_step"] = int(step)
    if first or prev < 0 or int(step) <= prev:     # 初回 / 逆行は減衰させない
        return
    elapsed = int(step) - prev
    half = cfg["half_life_steps"]
    drop = float(cfg["drop_threshold"])
    factor: dict[str, float] = {}
    for tier in TIERS:
        hl = int(half.get(tier, 0))
        factor[tier] = 1.0 if hl <= 0 else 0.5 ** (elapsed / float(hl))
    for node in sorted(st["nodes"]):
        per = st["nodes"][node]
        for kind in sorted(per):
            f = factor.get(tier_of(cfg, kind), 1.0)
            if f >= 1.0:                           # persistent = 減らさない
                continue
            per[kind]["s"] = float(per[kind]["s"]) * f
        gone = [k for k in sorted(per) if float(per[k]["s"]) < drop]
        for kind in gone:
            del per[kind]
            st["evaporated"] += 1
        if not per:
            del st["nodes"][node]


def phase(sim, step: int, sim_min: int) -> None:
    """step 末の単一作用点: 集約 + 蒸発。**既定 OFF は即 return**。

    ★世界状態(所持金・位置・関係・drive・opinion)は 1 つも動かさない。
      触るのは「場所に何の跡が残っているか」だけ。
    """
    if not enabled(sim):
        return
    cfg = cfg_of(sim)
    st = _state(sim)
    events = _new_events(sim)
    if cfg["kinds"]:
        budget = int(cfg["max_per_step"])
        for e in events:
            if budget <= 0:
                break
            if e.kind in SRC_TO_KIND and _aggregate(sim, cfg, st, e, step, sim_min):
                budget -= 1
    _evaporate(sim, cfg, st, step, sim_min)


# --------------------------------------------------------------------------- #
# 観測(後から来た者への 1 行)= 本 module で唯一プロンプトに触れる場所
# --------------------------------------------------------------------------- #
def line_for(sim, agent) -> str | None:
    """その場所に残っている痕跡の 1 行(無ければ None)。**既定 OFF は常に None**。

    ★**最強 1 件だけ**(同点は種名昇順=決定論)。複数行にすると 25 万体のプロンプトが
      場所の履歴で埋まり、かつ「痕跡が多い条件」が文量から読めてしまう。
    ★**強度・件数・階層名は 1 文字も出さない**(no-fingerprint)。出るのは
      ``sentence(kind)`` の定型 1 行だけ。
    ★**当事者(集約時に同席していた者)には出さない**(自分の記憶と二重になる)。
    ★★観測は痕跡を 1 ミリも動かさない: 強度も同席者名簿も**書き換えない**
      (if-lane-research §1-3 の示唆 6。数える ``lines`` は観測側のタリーで、
      痕跡そのものではない)。
    """
    if not enabled(sim):
        return None
    st = getattr(sim, "_trace_state", None)
    if st is None:                                 # ON だが 1 件も集約が起きていない
        return None
    node = getattr(agent, "node", None)
    if not node:
        return None
    per = st["nodes"].get(str(node))
    if not per:
        return None
    cfg = cfg_of(sim)
    thr = float(cfg["line_threshold"])
    aid = int(agent.id)
    best: tuple[str, float] | None = None
    for kind in sorted(per):
        rec = per[kind]
        s = float(rec["s"])
        if s < thr:
            continue
        if aid in rec["w"]:                        # 当事者 = 伝聞で二重に読ませない
            continue
        if best is None or s > best[1]:            # 同点は種名昇順(sorted の先勝ち)
            best = (kind, s)
    if best is None:
        return None
    line = sentence(best[0])
    if not line:
        return None
    st["lines"] += 1
    _bump(st["lines_by_kind"], best[0])
    return line


# --------------------------------------------------------------------------- #
# 観測記録(summary.json の "traces" キー。**OFF ではキー自体を出さない**)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """``summary.json`` の ``traces`` キー(既定 OFF は None = キー自体を出さない)。

    ★閲覧イベントは L1 に出さない(受動観測で毎 step 大量発生するため)。代わりに
      「何行がプロンプトへ入ったか」の総数と種別内訳をここに残す = アブレーション軸として
      事前登録するときの分母になる観測量。
    """
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    out: dict = {"schema": SCHEMA, "kinds": list(cfg["kinds"]),
                 "tier_of": {k: tier_of(cfg, k) for k in sorted(cfg["kinds"])},
                 "half_life_steps": {t: int(cfg["half_life_steps"].get(t, 0))
                                     for t in TIERS},
                 "line_threshold": float(cfg["line_threshold"])}
    st = getattr(sim, "_trace_state", None)
    if st is None:                                 # ON だが 1 件も源イベントが起きていない
        out.update({"marks": 0, "evaporated": 0, "lines_injected": 0,
                    "by_kind": {}, "lines_by_kind": {},
                    "live_nodes": 0, "live_traces": 0, "strength_max": 0.0})
        return out
    strengths = [float(r["s"]) for per in st["nodes"].values()
                 for r in per.values()]
    out.update({
        "marks": int(st["marks"]), "evaporated": int(st["evaporated"]),
        "lines_injected": int(st["lines"]),
        "by_kind": {k: int(v) for k, v in sorted(st["by_kind"].items())},
        "lines_by_kind": {k: int(v) for k, v in sorted(st["lines_by_kind"].items())},
        "live_nodes": len(st["nodes"]), "live_traces": len(strengths),
        "strength_max": (round(max(strengths), 4) if strengths else 0.0)})
    return out


def strength_of(sim, node: str, kind: str) -> float:
    """その node のその種の強度(**読むだけ**・state も属性も生やさない)。"""
    st = getattr(sim, "_trace_state", None)
    if st is None:
        return 0.0
    rec = (st["nodes"].get(str(node)) or {}).get(str(kind))
    return 0.0 if rec is None else float(rec["s"])
