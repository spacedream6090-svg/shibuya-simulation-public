"""群のオントロジー(文化圏×経験の世界観共有群。2026-07-21。ユーザー承認済み)。

外部アドバイス「地震対処に慣れた人と、地震をほとんど経験しない旅行者では行動が大きく異なる。
文化圏×経験で世界観を共有する群の割合をコントロールできると精密な予測が可能になる」を受け、
各エージェントに「文化圏×経験」の群(ontology group)を1つ割り当てる層。

方針(ユーザーの nature-like 志向): 行動分岐はルールで書かない=LLM に委ねる。プロンプトへは
群の「経験の事実」1行(config の groups.*.line)だけを注入し、条件分岐は一切書かない。行動差は
その事実を読んだ LLM が自ら生む。

★ このモジュールを src/society/ 直下(tests/test_contracts の CHECKED_DIRS 外)に置く理由:
  群の定義(文化圏名・経験水準の自然文)は config 由来のテキストを保持・整形するが、engine/
  cognition/world からは薄い呼び出し(割当と1行取得)だけを見せる。opinion.py / worldview.py と
  同じ隔離作法。基盤コードに文化名リテラルは書かない(すべて config から読む=データ駆動)。

決定論(乱数ゼロ・RngHub 無風):
  群割当 = stable hash(ontology.seed, persona id)の一様値 [0,1) を、層(tier=P3 presence キー)別
  composition の比率で分割して選ぶ。hashlib 自前=RngHub の既存 stream を一切引かない(乱数列に
  無風)。run.seed 非依存=**同一設定なら全ランで同一人物は同一群**(比較実験の要)。プールの再来街
  (P3 DormantStore hydrate)でも純関数=同一割当。

直交性(R1 doctrine #4): traits・k 条件を一切読まない=群割当は persona id のみの関数(因子と直交)。

既定 OFF(enabled=false)= 属性なし・プロンプト不変=ゴールデン L1 バイト一致。ON でも LLM 呼数・
発火・乱数列は不変(プロンプト内容のみ変化)。
"""
from __future__ import annotations

import hashlib

# tier(composition のキー)は P3 presence/pool の実際の層名に対応する:
#   resident / duty / workday_shift / cadence / stochastic(world/presence.py の presence キー)
#   + default(プール非使用の直接ラン=run.n_agents 生成用)。
_DEFAULT_TIER = "default"

# worldview 初期オフセット(S-b)の起点とクリップ範囲(worldview.py 準拠: 全員 0.5 起点)。
_WV_BASE = 0.5
_WV_CLIP = (0.02, 0.98)


def build_cfg(raw: dict | None) -> dict:
    """conf の ontology ブロックを正準化する(既定 OFF=属性なし=現行挙動と完全同一)。

    groups.*: {label, line, wv_offsets?, その他メタ(quake_exp/city_exp/ja 等はそのまま保持)}。
    composition.<tier>: {group_id: 比率}(合計は正規化されるので厳密に 1.0 でなくてよい)。
    """
    raw = dict(raw or {})
    cfg: dict = {
        "enabled": bool(raw.get("enabled", False)),
        "seed": int(raw.get("seed", 0)),
        "groups": {},
        "composition": {},
    }
    for gid, g in (raw.get("groups") or {}).items():
        g = dict(g or {})
        entry: dict = {"label": str(g.get("label", gid))}
        entry["line"] = str(g["line"]) if g.get("line") else ""
        wvo = g.get("wv_offsets") or {}
        if wvo:
            entry["wv_offsets"] = {str(k): float(v) for k, v in dict(wvo).items()}
        for k, v in g.items():                       # 観測・分析用メタ(quake_exp 等)を素通し
            if k not in ("label", "line", "wv_offsets"):
                entry[k] = v
        cfg["groups"][str(gid)] = entry
    for tier, mix in (raw.get("composition") or {}).items():
        cfg["composition"][str(tier)] = {
            str(k): float(v) for k, v in dict(mix or {}).items()}
    # ---- 直交する第2軸(方式B。文化圏軸と真に独立)。軸ごとに seed_offset で独立ハッシュ ----
    # 各軸: groups(label/line + 素通しメタ)/ composition.<tier>(age 不明時の全体分布)/
    #        bands.<age_lo>(年代条件付き構成比。agent.age がある時に優先)/ seed_offset。
    # 既定 OFF・軸未指定でも空 dict=現行と割当・注入とも完全同一(assign_group は axes を読まない)。
    # 各軸: groups / composition.<tier> / bands.<age_lo>(年代条件付き) / seed_offset に加え、
    #        party_map(実人数→群 id のデータ駆動写像=ハッシュ不要)/ day_varying(準静的=day を混ぜる)。
    cfg["axes"] = {}
    for name, ax in (raw.get("axes") or {}).items():
        ax = dict(ax or {})
        axc: dict = {"seed_offset": int(ax.get("seed_offset", 0)),
                     "groups": {}, "composition": {}, "bands": {},
                     "party_map": {}, "day_varying": bool(ax.get("day_varying", False))}
        for gid, g in (ax.get("groups") or {}).items():
            g = dict(g or {})
            e: dict = {"label": str(g.get("label", gid))}
            e["line"] = str(g["line"]) if g.get("line") else ""
            for k, v in g.items():                    # 観測・分析用メタを素通し
                if k not in ("label", "line"):
                    e[k] = v
            axc["groups"][str(gid)] = e
        for tier, mix in (ax.get("composition") or {}).items():
            axc["composition"][str(tier)] = {
                str(k): float(v) for k, v in dict(mix or {}).items()}
        for lo, mix in (ax.get("bands") or {}).items():
            axc["bands"][int(lo)] = {
                str(k): float(v) for k, v in dict(mix or {}).items()}
        # party_map: 実人数(文字列キー "1"/"2"…)→群 id。'default' キーは上限超(例 3人以上)の受け皿。
        # キーを文字列に統一する(混在型キーだと config_hash の json.dumps(sort_keys=True) が壊れる)。
        for k, v in (ax.get("party_map") or {}).items():
            axc["party_map"][str(k)] = str(v)
        cfg["axes"][str(name)] = axc
    # ---- 訓練経験行(方式A)。文化圏群の drill_line を別行として付加(既定 line は不変)----
    cfg["drill_enabled"] = bool((raw.get("drill") or {}).get("enabled", False))
    return cfg


def _stable_uniform(seed: int, pid: str) -> float:
    """(seed, persona id)から run.seed 非依存の一様値 [0,1) を作る(hashlib=決定論・乱数列無風)。

    hashlib はプロセス跨ぎで安定(PYTHONHASHSEED の salt を受けない)=リプレイ・resume・別ランで
    同一 pid は常に同一値=同一群。RngHub の stream は一切引かない。"""
    h = hashlib.blake2b(f"{int(seed)}\x1f{pid}".encode("utf-8"),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _choose(comp: dict, u: float) -> str | None:
    """composition(group_id→比率)と一様値 u∈[0,1) から群 id を1つ選ぶ(group id 昇順で累積分割)。

    dict 順に依らず同一 comp・同一 u で同一割当。総和 0 は None。数値誤差の保険で最後の群に落とす。"""
    items = sorted((gid, max(0.0, float(w))) for gid, w in comp.items())
    total = sum(w for _, w in items)
    if total <= 0.0:
        return None
    r = u * total
    cum = 0.0
    for gid, w in items:
        cum += w
        if r < cum:
            return gid
    return items[-1][0]                               # 数値誤差の保険(必ず最後の群に落とす)


def assign_group(cfg: dict, pid, tier: str) -> str | None:
    """persona id と層(tier)から群 id を決定論で割り当てる(乱数ゼロ・k/trait 非参照)。

    tier は P3 presence キー(resident/duty/workday_shift/cadence/stochastic)または直接ラン用
    default。未知 tier は default に後退。composition[tier] の比率を **group id 昇順** で累積し、
    _stable_uniform で分割して選ぶ(dict 順に依らず同一 cfg で同一割当)。
    """
    if not cfg.get("enabled"):
        return None
    comp = cfg["composition"].get(str(tier))
    if comp is None:
        comp = cfg["composition"].get(_DEFAULT_TIER)
    if not comp:
        return None
    return _choose(comp, _stable_uniform(int(cfg["seed"]), str(pid)))


def _band_for_age(bands: dict, age) -> dict | None:
    """年齢に対応する年代 band(下限キーが age 以下で最大のもの)の構成比を返す。該当なしは None。"""
    chosen = None
    for lo in sorted(bands):
        if int(age) >= int(lo):
            chosen = bands[lo]
        else:
            break
    return chosen


def _party_bucket(pmap: dict, size) -> str | None:
    """実人数 party_size を群 id に写像する(1→solo/2→pair/3+→'default'群=データ駆動・ハッシュ不要)。

    正の整数のみ扱う。厳密一致キー(int)優先、無ければ 'default'(上限超の受け皿)。0/負/非数は
    None を返し呼び出し側の composition 後退に委ねる(=party_size を持たない個体と同じ扱い)。"""
    try:
        n = int(size)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    key = str(n)                                         # party_map キーは文字列(build_cfg で統一)
    if key in pmap:
        return pmap[key]
    return pmap.get("default")


def assign_axis(cfg: dict, axis_name: str, pid, tier: str, age=None,
                day=None, party_size=None) -> str | None:
    """第2軸以降(方式B)の群割当。文化圏軸と独立な一様値(seed + seed_offset)で決定論割当。

    party_size(実人数)がある個体は party_map で**データ駆動**に確定(ハッシュ不要=最優先・固定)。
    無い個体は age があれば bands で年代条件付き構成比を優先、無ければ composition[tier](未知 tier は
    default)= 全体分布を hashlib で分割。day_varying=true の軸は (pid, day) の純関数(day を混ぜて
    日ごとに変わるが決定論=同一 day・同一 pid は同一群=resume/hydrate 安定)。traits/k は一切読まない
    =R1 直交。seed_offset により他軸(文化圏=seed のみ)ハッシュと無相関=軸間の割当は独立(直交)。"""
    if not cfg.get("enabled"):
        return None
    axis = (cfg.get("axes") or {}).get(str(axis_name))
    if not axis:
        return None
    pmap = axis.get("party_map")                          # party_size 由来=最優先・固定(day 非依存)
    if pmap and party_size is not None:
        gid = _party_bucket(pmap, party_size)
        if gid is not None and gid in axis["groups"]:
            return gid
    comp = None
    if age is not None and axis.get("bands"):
        comp = _band_for_age(axis["bands"], age)
    if comp is None:
        comp = axis["composition"].get(str(tier))
        if comp is None:
            comp = axis["composition"].get(_DEFAULT_TIER)
    if not comp:
        return None
    off = int(axis.get("seed_offset", 0))
    key = str(pid)
    if axis.get("day_varying") and day is not None:      # 準静的: (pid, day) 純関数=日ごとに変わる
        key = f"{pid}\x1f{int(day)}"
    return _choose(comp, _stable_uniform(int(cfg["seed"]) + off, key))


def group_line(cfg: dict, gid: str | None) -> str | None:
    """群 id → プロンプトに注入する「経験の事実」1行(config の groups.*.line)。無ければ None。"""
    if not gid:
        return None
    g = cfg["groups"].get(gid)
    if not g:
        return None
    line = g.get("line")
    return str(line) if line else None


def axis_line(cfg: dict, axis_name: str, gid: str | None) -> str | None:
    """第2軸以降の群 id → プロンプトに注入する「一人称の行動の事実」1行(axes.*.groups.*.line)。"""
    if not gid:
        return None
    axis = (cfg.get("axes") or {}).get(str(axis_name))
    if not axis:
        return None
    g = axis["groups"].get(gid)
    if not g:
        return None
    line = g.get("line")
    return str(line) if line else None


def group_drill_line(cfg: dict, gid: str | None) -> str | None:
    """文化圏群の防災訓練経験1行(方式A・別キー drill_line)。drill.enabled=false/未設定なら None。

    既定 line は変更せず、訓練経験を独立の1行として付加する(割当は文化圏群のまま=不変)。"""
    if not gid or not cfg.get("drill_enabled"):
        return None
    g = cfg["groups"].get(gid) or {}
    dl = g.get("drill_line")
    return str(dl) if dl else None


def initial_controllability(cfg: dict, gid: str | None) -> float | None:
    """群 id → worldview 可制御性の初期値(S-b)。offset 無指定/0 のときは None(=既定 0.5 のまま)。

    全員 0.5 起点に群の小さな初期オフセットを1回だけ載せる決定論値。以後の更新則は worldview 側の
    従来どおり(経験による経路依存を保存)=ここは起点だけをずらす。"""
    if not gid:
        return None
    g = cfg["groups"].get(gid) or {}
    dc = (g.get("wv_offsets") or {}).get("controllability")
    if not dc:
        return None
    lo, hi = _WV_CLIP
    return min(hi, max(lo, _WV_BASE + float(dc)))
