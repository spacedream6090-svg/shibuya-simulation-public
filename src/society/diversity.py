"""観光・多言語・犯罪・治安(現実ギャップ 後続波 H5 2026-07-07。ユーザー要望)。

渋谷の街の多様性と負の相互作用を最小・非LLM・決定論で載せる層。4機構:

  1. 観光客・訪日客(assign_attributes / tourist_dest): visitor の一部を「回遊型の観光客」に
     決定論(名簿・id 順)で仕立て、自由時間の行き先を landmark_score 高ノード(イメージアビリティ、
     既存 Lynch と同素材)へ寄せる。ランドマーク回遊=tourist_visit(専用 stream "tourist")。
     プロンプト文脈「観光で来ている」。
  2. 多言語・文化の多様性(assign_attributes / cross_barrier / context_line): 一部エージェントを
     非日本語話者に決定論で仕立て、語の採用(labeling)で異言語間は採用閾値を上げる(伝播障壁)。
     プロンプト文脈「主に○○語を話す」。伝播障壁は labeling 層(既存の採用ロジック)に載せる。
  3. 犯罪・迷惑行為(tick_crime): 街内で窃盗・迷惑行為の負の相互作用(現状皆無)を発生させる。
     窃盗=被害者の所持金−(+ grievance を factors 経由)、迷惑=その場の周囲の grievance。
     crime / nuisance を記録する。近傍(同ノード)に警察官(G3)が居れば抑止。専用 stream "crime"。
  4. 治安の体感・回避行動(is_danger / safe_dests): 犯罪履歴のあるノード=危険地帯を、自由時間の
     行き先(EPR)から避ける(空間行動の変化)。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  観光回遊・言語属性・犯罪発生・危険地帯回避のロジック(grievance の factors hook 呼び出しを含む)を
  ここに閉じる=engine/cognition/world に構成概念名を書かない(no-fingerprint 契約に触れない)。
  grievance は factors 層 hook(on_crime)へ**不透明な magnitude** だけを渡す。窃盗の money− は
  観測可能な客観量(現金)の増減であって構成概念ではない(economy と同じ扱い)。

R1 呼数不変: どの機構も generate() を1本も足さない。観光属性・言語属性は名簿からの決定論付与、
  観光回遊は専用 stream "tourist"、犯罪発生は専用 stream "crime"(いずれも既存 draw 順に挿入しない)、
  伝播障壁は名簿の言語属性からの純関数、危険地帯回避は犯罪履歴(観測量)からの決定論。観光回遊・犯罪
  (被害者 money−)・危険地帯回避は物理位置=対面 co-location を変えうる(=FixedLLM で ON!=OFF になりうる=
  career G5 / crowd G4 / 健康 H1 / 世帯 H2 / 商業 H3 / 災害 H4 と同型)が、機構は k・内面状態(構成概念)を
  発火判断に食わせず名簿・config・新 stream・物理位置・犯罪履歴(k 非依存)のみ参照する=compute_matched
  下の k 不変性(k=free==k=off の呼数一致)で担保する。grievance は drive(発火系)には接続しない。

★路上の生業(Wave 4 III-3 = src/society/street_life.py)との関係(2 点。既定 OFF では無風):
  1. **迷惑行為「客引き」の重複解消**: street_life が ON のとき、tick_crime は nuisance の種類から
     「客引き」を**供給しない**(nuisance_kinds_for)。あちらでは客引きが「夜の店に雇われた従業員が
     条例の禁止行為として行うこと」= 主体・場所・罰則(過料5万円)を持つ行為になるので、同じ出来事を
     無主体の確率イベントとして二重に出すと条例の効き目が測れなくなる(併存ではなく**置き換え**)。
  2. **路上生活者の尊厳規約**: street_life が ON のとき、路上生活者は加害者・被害者・周囲の
     **すべて**から除外される(_street_life_excluded)。これは現実の主張ではなく「この機構と
     結びつけない」という設計判断で、L1 の crime/nuisance に彼らの id が 1 度も現れないことを保証する。

既定 OFF(enabled=false)= 観光属性・言語属性を付けず・観光回遊なし・伝播障壁なし・犯罪/迷惑なし・
  危険地帯回避なし・"tourist"/"crime" stream も引かない=イベント 0 件・乱数消費不変(ゴールデン
  golden_baseline_l1.json を守る)。新イベント種は crime / nuisance / tourist_visit(schema.py 登録済み=編集しない)。
"""
from __future__ import annotations

from collections import Counter, defaultdict

from . import economy_sfc as sfc_mod
from .factors import update as factor_update
from .observer.schema import Event

# 警察官(G3 執行の公務員)。近傍に居れば犯罪を抑止する。scheduler の POLICE_OCCS と同一
# (循環 import を避けるためここでも定義する。src/society 直下)。
POLICE_OCCS = ("警察官",)


DEFAULTS = {
    "enabled": False,
    # ---- 観光客・訪日客(回遊型)----
    "tourist_ratio": 0.0,        # ★ON 推奨: 例 0.5(visitor のうち観光客にする割合。決定論・id 順)
    "tourist_bias": 0.7,         # 自由時間にランドマークへ回遊する確率/step(stream "tourist")
    "landmark_top": 8,           # 回遊先候補にする上位ランドマーク数
    # ---- 多言語・文化の多様性 ----
    "foreign_ratio": 0.0,        # ★ON 推奨: 例 0.2(非日本語話者にする割合。決定論・id 順)
    "languages": ["英語", "中国語", "韓国語"],   # 割り当てる言語ラベル(循環)
    "cross_lang_barrier": 2,     # 異言語間の語採用に必要な追加聴取回数(伝播障壁)
    # ---- 犯罪・迷惑行為 ----
    "crime_prob": 0.0,           # ★ON 推奨: 例 0.005(街内の1人1stepあたりの窃盗発生確率。stream "crime")
    "nuisance_prob": 0.0,        # ★ON 推奨: 例 0.005(迷惑行為の発生確率。stream "crime")
    "theft_amount": 3000.0,      # 窃盗の被害額(被害者 money−。現金の範囲で)
    "crime_grievance": 0.0,      # ★ON 推奨: 例 0.08(窃盗被害者の不満増分。factors 経由。0.0=金銭のみ)
    "nuisance_grievance": 0.0,   # ★ON 推奨: 例 0.03(迷惑行為の周囲の不満増分。factors 経由)
    "nuisance_kinds": ["客引き", "ナンパ", "喧嘩", "路上の騒ぎ"],
    # ---- 治安の体感・回避 ----
    "danger_threshold": 3,       # 犯罪履歴回数がこれ以上のノード=危険地帯(避ける)。0 以下=回避無効
    "avoid_danger": True,        # 危険地帯を自由時間の行き先から避ける(enabled 時のみ効く)
}

_BOOL_KEYS = ("enabled", "avoid_danger")
_INT_KEYS = ("landmark_top", "cross_lang_barrier", "danger_threshold")
_FLOAT_KEYS = ("tourist_ratio", "tourist_bias", "foreign_ratio", "crime_prob",
               "nuisance_prob", "theft_amount", "crime_grievance", "nuisance_grievance")
_LIST_STR_KEYS = ("languages", "nuisance_kinds")


def build_cfg(raw) -> dict:
    """conf の society_diversity ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(career/health/household/commerce/disaster と同型)。
    すべて既定 OFF(enabled=false)で simulation の属性付与・scheduler の犯罪フェーズ・routine の
    観光回遊/危険地帯回避・labeling の伝播障壁・deliberate の文脈注入が完全 no-op(イベント 0 件・
    新 stream も引かない=ゴールデンを守る)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    cfg = dict(DEFAULTS)
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _INT_KEYS:
            cfg[k] = int(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
        elif k in _LIST_STR_KEYS:
            cfg[k] = [str(x) for x in v]
    return cfg


def enabled(sim) -> bool:
    """観光・多言語・犯罪・治安(H5)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "diversitycfg", None)
    return bool(cfg and cfg["enabled"])


# ---------------------------------------------------------------- 属性付与(起動時1回)
def assign_attributes(sim) -> None:
    """起動時1回: 観光客型 visitor と非日本語話者を決定論(id 昇順)で仕立てる。

    OFF は完全 no-op(属性を付けない=getattr 既定で不変)。決定論(乱数を一切引かない=既存 draw 順に
    影響しない)。観光客は visitor の先頭 tourist_ratio 割、非日本語話者は全体の先頭 foreign_ratio 割
    (id 昇順)。属性は agent の動的属性(tourist / language)として付す=agent.py を変えない
    (health/household と同型の getattr 既定でバイト一致)。"""
    if not enabled(sim):
        return
    cfg = sim.diversitycfg
    agents = sorted(sim.agents, key=lambda a: a.id)
    # 観光客: visitor の一部を回遊型に(名簿/決定論)
    visitors = [a for a in agents if a.visitor]
    n_tour = int(round(float(cfg["tourist_ratio"]) * len(visitors)))
    for i, a in enumerate(visitors):
        a.tourist = bool(i < n_tour)
    # 多言語: 一部を非日本語話者に(決定論・言語ラベルは循環割当)
    langs = cfg["languages"]
    n_for = int(round(float(cfg["foreign_ratio"]) * len(agents)))
    if langs:
        for i, a in enumerate(agents):
            if i < n_for:
                a.language = langs[i % len(langs)]


def context_line(agent, place_name: str = "この街") -> str | None:
    """発火/計画/内省プロンプト用: 観光・言語の文脈行(決定論・k 非依存)。

    観光客なら「観光で来ている」、非日本語話者なら「主に○○語を話す」を添える。該当なしなら None
    (build_prompt は1行も足さない=OFF/非該当はバイト一致)。place_name は街名(envpack)。"""
    parts: list[str] = []
    if getattr(agent, "tourist", False):
        parts.append(f"あなたは観光で{place_name}を訪れている(この街の生活者ではなく、名所を見て回っている)。")
    lang = getattr(agent, "language", "")
    if lang:
        parts.append(f"あなたは主に{lang}を話し、日本語での会話は少し不自由に感じている。")
    return " ".join(parts) if parts else None


# ---------------------------------------------------------------- 多言語の伝播障壁(labeling 層)
def cross_barrier(sim, speaker_id: int, listener) -> int:
    """話者と聞き手の言語が異なるときの語採用の追加閾値(伝播障壁)。同言語/OFF なら 0(不変)。

    labeling(LabelSystem.on_hear)の extra_threshold に渡す不透明な整数。話者不明(media=-1 等)は
    日本語(既定)扱い。OFF は常に 0=on_hear が従来と完全同一(バイト一致)。"""
    if not enabled(sim):
        return 0
    speaker = sim.agent_by_id.get(speaker_id) if speaker_id is not None else None
    s_lang = getattr(speaker, "language", "") if speaker is not None else ""
    l_lang = getattr(listener, "language", "")
    return int(sim.diversitycfg["cross_lang_barrier"]) if s_lang != l_lang else 0


# ---------------------------------------------------------------- 観光客のランドマーク回遊
def _landmarks(sim) -> list[str]:
    """回遊先の上位ランドマーク(イメージアビリティ)。sim に一度だけキャッシュする。

    既存の Lynch landmark_score があればそれを使い、無ければ POI 密度+命名の有無から同素材の
    スコアを組む(Lynch と同じ考え方=決め打ちの landmark 指定はしない)。スコア降順・id 昇順で
    上位のみを候補にする(landmark_top)。研究者frame の空間データのみ参照=因子名を見ない。"""
    cache = getattr(sim, "_diversity_landmarks", None)
    if cache is not None:
        return cache
    score = getattr(sim, "landmark_score", None)
    if not score:
        score = {}
        for node in sim.dests:
            n_pois = len(sim.city.pois_at_node(node))
            named = 1.0 if sim.city.graph.nodes[node].get("name") else 0.0
            n_named_bldg = sum(1 for b in sim.city.buildings_at(node) if b.get("name"))
            score[node] = float(n_pois) + named + float(n_named_bldg)
    ranked = sorted(score.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    top = [n for n, s in ranked if float(s) > 0.0] or [n for n, _ in ranked]
    top = top[: max(1, int(sim.diversitycfg["landmark_top"]))]
    sim._diversity_landmarks = top
    return top


def tourist_dest(agent, sim, step: int, sim_min: int) -> str | None:
    """観光客の自由時間の行き先(ランドマークへの回遊)。決定論・非LLM。tourist_visit を記録する。

    OFF / 非観光客 / 抽選外 / 候補なし なら乱数を一切引かず None(既定不変)。専用 stream "tourist"
    (既存 draw 順を汚さない)で tourist_bias 抽選し、上位ランドマークから決定論で1つ選ぶ。現在地と
    同じなら None(=stay)。移動(行き先)だけを変え、発火判断・LLM 呼数は増やさない(R1)。"""
    if not enabled(sim):
        return None
    if not getattr(agent, "tourist", False):
        return None
    rng = sim.hub.stream("tourist", agent.id, step)
    if rng.random() >= float(sim.diversitycfg["tourist_bias"]):
        return None
    cands = [n for n in _landmarks(sim) if n != agent.node]
    if not cands:
        return None
    node = cands[int(rng.integers(len(cands)))]
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="tourist_visit", x=agent.x, y=agent.y,
                         payload={"node": node}))
    return node


# ---------------------------------------------------------------- 治安の体感・回避
def _crime_nodes(sim) -> Counter:
    """ノード別の犯罪履歴カウンタ(危険地帯の判定材料)。sim に一度だけ用意する。"""
    cn = getattr(sim, "_diversity_crime_nodes", None)
    if cn is None:
        cn = Counter()
        sim._diversity_crime_nodes = cn
    return cn


def note_crime_node(sim, node: str) -> None:
    """そのノードの犯罪履歴を 1 件積む(危険地帯 = 治安回避の材料)。

    ★H4(``incidents_interpersonal``)が窃盗の世代交代で出した事件も、従来の窃盗と
      **同じ履歴**に積まれるようにするための公開口(あちらから private を触らせない)。
      本 module が OFF でも安全: ``is_danger`` / ``safe_dests`` が OFF なら常に無効化
      されるので、カウンタが積まれても行き先は 1 つも変わらない。
    """
    _crime_nodes(sim)[str(node)] += 1


def is_danger(sim, node: str) -> bool:
    """このノードが危険地帯(犯罪履歴 ≥ danger_threshold)か。OFF / 閾値0以下は常に False。"""
    if not enabled(sim):
        return False
    thr = int(sim.diversitycfg["danger_threshold"])
    if thr <= 0:
        return False
    return _crime_nodes(sim)[node] >= thr


def safe_dests(sim, agent, dests: list[str]) -> list[str]:
    """自由時間(EPR)の行き先候補から危険地帯を除いたリスト。OFF / 回避無効 なら dests をそのまま返す。

    現在地は常に残す(移動しない選択を潰さない)。全て危険で空になるなら元の dests に後退する
    (行き先を失わない)。OFF は同一リストを返す=choose_destination の挙動がバイト一致。"""
    if not enabled(sim) or not sim.diversitycfg["avoid_danger"]:
        return dests
    thr = int(sim.diversitycfg["danger_threshold"])
    if thr <= 0:
        return dests
    cn = _crime_nodes(sim)
    safe = [d for d in dests if cn[d] < thr or d == agent.node]
    return safe or dests


# ---------------------------------------------------------------- 犯罪・迷惑行為(毎step)
def _pick_victim(here: list, offender, exclude: frozenset = frozenset()):
    """同ノードの被害者(自分・警察官・尊厳規約の除外 id を除く最小 id)。居なければ None。"""
    cands = sorted((o for o in here if o.id != offender.id
                    and o.occupation not in POLICE_OCCS
                    and int(o.id) not in exclude), key=lambda o: o.id)
    return cands[0] if cands else None


# ---------------------------------------------------------------- 路上の生業(Wave 4 III-3)との接続
# 迷惑行為の「客引き」= street_life の ``touting`` と同じ出来事(下の nuisance_kinds_for を参照)
_TOUTING_KIND = "客引き"


def _street_life_excluded(sim) -> frozenset:
    """犯罪・迷惑機構から**除外する** id 集合(``street_life`` の尊厳規約 2)。

    ★``street_life`` が OFF(既定)なら必ず空集合 = 下のフィルタが完全 no-op になり、
      ゴールデン L1 バイト一致が壊れない(ON でも 路上生活者が名簿に居なければ空)。
    ★これは「路上生活者は犯罪をしない」という現実の主張ではなく、**この機構と
      結びつけない**という設計判断である(結びつければシムがスティグマを再生産する)。
      除外は加害者・被害者・周囲の**すべて**に掛ける = L1 の crime / nuisance の
      payload に彼らの id が 1 度も現れない(tests/test_street_life.py が機械固定)。
    """
    from . import street_life as street_life_mod
    return street_life_mod.rough_sleeper_ids(sim)


def _theft_superseded(sim) -> bool:
    """窃盗の枝を H4(``incidents_interpersonal``)へ**明け渡す**か。

    ★**既定 OFF では必ず False** = 下の窃盗の枝が 1 バイトも変わらない
      (``_street_life_excluded`` が OFF で空集合を返すのと同じ「二重に保守的」な形)。
    """
    from . import incidents_interpersonal as incidents_mod
    return incidents_mod.superseded_theft(sim)


def nuisance_kinds_for(sim, cfg: dict) -> list:
    """この step に使う迷惑行為の種類(``street_life`` ON なら「客引き」を**供給しない**)。

    ★重複の解消(単一の源): 客引きは ``street_life`` が ON のとき「夜の店に雇われた
      従業員が条例の禁止行為として行うこと」= **主体と場所と罰則を持つ行為**になる。
      同じ出来事を「無主体の確率イベント(nuisance)」として二重に出すと、L1 の
      客引き件数が 2 つの機構の和になり、条例の効き目(``touting_fine`` 後の減少)が
      測れなくなる。そこで ON のときだけ ``nuisance_kinds`` から客引きを外す
      (=**併存ではなく置き換え**)。他の 3 種(ナンパ・喧嘩・騒ぎ)は無関係なので残る。
    ★OFF(既定)では cfg の値をそのまま返す = 従来と完全同一(バイト一致)。
    """
    kinds = cfg["nuisance_kinds"] or ["迷惑行為"]
    from . import street_life as street_life_mod
    if not street_life_mod.enabled(sim):
        return kinds
    out = [k for k in kinds if k != _TOUTING_KIND]
    return out or ["迷惑行為"]


def tick_crime(sim, step: int, sim_min: int) -> None:
    """毎step: 街内で窃盗・迷惑行為を発生させる(専用 stream "crime")。既定 OFF=完全 no-op(バイト一致)。

    位置が確定した後(scheduler の _phase_move / _phase_enforcement の後)に呼ぶ。co-location(同ノード)の
    非公務員が offender 候補。同ノードに警察官が居れば抑止(G3 執行との接続)。1人1step に stream "crime"
    から1回だけ引き、[0,crime_prob)=窃盗 / [crime_prob,crime_prob+nuisance_prob)=迷惑 に振り分ける
    (相互排他・決定論)。窃盗=被害者 money−(現金の範囲で)+ grievance(factors 経由・不透明 magnitude)、
    迷惑=同ノードの周囲の grievance。crime / nuisance を記録し、犯罪履歴(危険地帯)を積む。
    grievance は drive(発火系)に接続しない(R1: 呼数を1本も動かさない)。"""
    if not enabled(sim):
        return
    cfg = sim.diversitycfg
    cp = float(cfg["crime_prob"])
    npb = float(cfg["nuisance_prob"])
    if cp <= 0.0 and npb <= 0.0:
        return
    at_node: dict[str, list] = defaultdict(list)
    police_nodes: set[str] = set()
    for a in sim.agents:
        if a.loc == "outside" or a.sleeping:
            continue
        at_node[a.node].append(a)
        if a.occupation in POLICE_OCCS:
            police_nodes.add(a.node)
    theft_amt = float(cfg["theft_amount"])
    cg = float(cfg["crime_grievance"])
    ng = float(cfg["nuisance_grievance"])
    kinds = nuisance_kinds_for(sim, cfg)
    excluded = _street_life_excluded(sim)                      # 尊厳規約 2(OFF は空集合)
    theft_superseded = _theft_superseded(sim)                  # H4 世代交代(OFF は必ず False)
    cn = _crime_nodes(sim)
    for a in sorted(sim.agents, key=lambda a: a.id):           # offender は id 昇順=決定論
        if a.loc == "outside" or a.sleeping:
            continue
        if a.occupation in POLICE_OCCS:                        # 警察官は加害しない
            continue
        if int(a.id) in excluded:                              # 尊厳規約 2: 加害者にしない
            continue
        if a.node in police_nodes:                             # 近傍(同ノード)警察官で抑止
            continue
        r = sim.hub.stream("crime", a.id, step).random()
        if cp > 0.0 and r < cp:                                # 窃盗
            # ★H4 世代交代(``incidents_interpersonal``。**既定 OFF では必ず False**):
            #   窃盗は「1人1step のレート抽選」から「共在ペアの上の条件付き確率」へ移った。
            #   併存させると L1 の crime 件数が 2 機構の和になり RAT の効き目が測れないので
            #   **置き換える**(``street_life`` ON で「客引き」の供給を止めたのと同じ作法)。
            #   ★抽選そのものは飛ばさない: ここで continue するのは**枝の中身だけ**で、
            #     stream "crime" の消費列は ON/OFF で 1 バイトも変わらない。
            if theft_superseded:
                continue
            victim = _pick_victim(at_node[a.node], a, excluded)
            if victim is None:
                continue
            stolen = min(float(victim.money), theft_amt)
            victim.money = max(0.0, float(victim.money) - theft_amt)
            cr_payload = {"kind": "theft", "victim": int(victim.id),
                          "offender": int(a.id),
                          "amount": round(stolen, 1)}
            # IF-E2 UNCOVERED(第98バッチ・既定 OFF=キーなし): 窃盗は相互合意が無いので
            # **取引ではない**(SNA 2008 §3.98)。RoW(取引の相手方)ではなく K5
            # (その他の資産変動)へ分類する。**加害者へは入金しない**(現行の動力学のまま)。
            # 渡すのは丸める前の実減少額(stolen)= 会計は payload の丸めに引きずられない。
            payee = sfc_mod.on_theft(sim, stolen)
            if payee is not None:
                cr_payload["payee"] = payee
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                                 kind="crime", x=a.x, y=a.y,
                                 payload=cr_payload))
            factor_update.on_crime(victim, cg, cause="crime", step=step,
                                   sim_min=sim_min, logger=sim.logger)
            victim.remember("何かを盗まれたようだ")
            cn[a.node] += 1
        elif npb > 0.0 and r < cp + npb:                       # 迷惑行為(客引き/ナンパ/喧嘩/騒ぎ)
            bystanders = [o for o in at_node[a.node]
                          if o.id != a.id and o.occupation not in POLICE_OCCS
                          and int(o.id) not in excluded]   # 尊厳規約 2: 周囲にも入れない
            if not bystanders:
                continue
            kind = kinds[(int(a.id) + step) % len(kinds)]
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                                 kind="nuisance", x=a.x, y=a.y,
                                 payload={"kind": kind, "node": a.node}))
            for o in bystanders:
                factor_update.on_crime(o, ng, cause="nuisance", step=step,
                                       sim_min=sim_min, logger=sim.logger)
            cn[a.node] += 1
