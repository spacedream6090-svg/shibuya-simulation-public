"""D16: チェックポイント(完全状態の保存)と途中再開。

設計原則
--------
- RngHub はステートレス(stream は (用途, id, step) から都度導出)。master seed だけで
  再現できるので**乱数状態は保存しない**。
- city(地図)/ router / clock / logger / llm / transit は config から再構築できる or
  load 後に再配線するので**保存しない**。scenario の封鎖(closed フラグ)だけは
  load 時に city.graph へ再適用し、router.invalidate() で経路キャッシュを捨てる。
- 「何を集め、どう戻すか」は本モジュールに一元管理する(個別クラスに __getstate__ を
  生やさない)。共有参照(labels.items ⇄ sim.items の ItemStore、Item 実体、
  labels.text_to_item と items.items が同じ Item を指すこと)は **1 回の
  pickle.dumps に同梱**して保つ(pickle は 1 回の dumps 内では共有参照を保存する)。
- 決定論: set は原則 membership / sorted 反復のみ(全ファイル監査済み)。順序安定の
  ため scenario.closed は sorted list で直列化して復元する。pickle はプロセス内でも
  set の反復順を保存しない(dict は保存する)ため、反復順が観測に効く箇所は sorted で
  なければならない — 監査の結果、そうなっている(唯一の生反復 adopted は全経路で
  sorted、例外は engine の post/dm フォールバック 1 行のみで mock では非到達)。
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path

FORMAT = 1

# resume で正当に変わりうるキー(整合性ハッシュから除外する)。
# n_steps は「さらに先まで回す」ため必ず変わる。name/out_dir/checkpoint_every も run 制御。
_VOLATILE_KEYS = [
    ("run", "n_steps"), ("run", "name"), ("run", "out_dir"),
    ("observer", "checkpoint_every"),
]


def config_hash_from_container(data: dict) -> str:
    """resolved config(OmegaConf.to_container 済み dict)の内容ハッシュ。

    resolve 済みの dict を既に持っている呼び出し側(observer/manifest.py)が
    to_container を二度走らせずに済むよう、定義をここ 1 箇所に保ったまま口を分けてある。
    """
    data = copy.deepcopy(data)
    for sect, key in _VOLATILE_KEYS:
        if isinstance(data.get(sect), dict):
            data[sect].pop(key, None)
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def config_hash(cfg) -> str:
    """決定論に効く config の内容ハッシュ(resume 制御キーは除外)。"""
    from omegaconf import OmegaConf
    return config_hash_from_container(OmegaConf.to_container(cfg, resolve=True))


def save(sim, step: int, path: str | Path) -> Path:
    """シミュレーションの完全状態を pickle+gzip で保存する(原子的 rename)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": FORMAT,
        "step": int(step),
        "config_hash": config_hash(sim.cfg),
        # --- 完全状態(自己完結・transient 参照を含まないオブジェクト群) ---
        "agents": sim.agents,             # dataclass/MemoryStore/Counter/set/opinion/money…
        "labels": sim.labels,             # LabelSystem(内部に ItemStore=sim.items を同梱)
        "net": sim.net,                   # posts/news/follows/contacts/likes/priority/既読
        "tools": sim.tools,               # events/groups/proposals/ventures/flyers/member_of
        "rulebook": sim.rulebook,         # 制度DSL: active ルール/日次進行/ブースト(自己完結)
        "world_events": sim.world_events, # 残り(shock_event が実行時に append しうる)
        # --- scheduler が sim に持つランタイムカウンタ ---
        "runtime": {
            "drive_stats": sim.drive_stats,
            "econ_day": sim._econ_day,
            "acct_day": sim._acct_day,          # 口座 E5: 暦日境界の進行(給料日/家賃)
            # 会社観測データ層 B4: per-day アキュムレータ(mid-day checkpoint でも resume==straight を保つ)。
            # OFF ランでは空 dict / -1=挙動不変(load は .get で旧 checkpoint 互換)。workers 集合は
            # emit 時に必ず sorted 反復=pickle の集合反復順非保存の影響を受けない(determinism 監査済み)。
            "org_day": getattr(sim, "_org_day", {}),
            "org_ledger_day": getattr(sim, "_org_ledger_day", -1),
            # 内部可動性 第60バッチ b: 転居/同棲の日境界進行(mid-day checkpoint でも resume==straight
            # を保つ)。新規の per-agent 状態(通勤既知値/同棲経過日/同棲フラグ)は agents pickle に自然
            # 同梱=ここは日カウンタのみ中央管理する。OFF ランでは -1=挙動不変(load は .get で旧 ckpt 互換)。
            "housing_day": getattr(sim, "_housing_day", -1),
            # キャリア転換(Wave G5)の日境界進行。第60バッチ b の career 由来転居が resume で転職を
            # 二重発火しないよう中央管理する(従来は未保存=career ON の mid-day resume が未検証だった)。
            # OFF ランでは -1=挙動不変(load は .get で旧 checkpoint 互換)。
            "career_day": getattr(sim, "_career_day", -1),
            # 社会関係(Wave G2)の日境界進行 第61検収補修: 未保存だと relations ON の mid-day resume で
            # _phase_relations_day が同じ日を再処理(closeness減衰/評判風化の二重発火)し resume!=straight。
            # 第61バッチの gossip resume テストが顕在化させた既存ギャップ。OFF ランでは -1=挙動不変。
            "rel_day": getattr(sim, "_rel_day", -1),
            # 資産レンズ 第59(検収補修): 前日 wealth スナップ+τ(observer/assets の state)。
            # processed はプロセス内 logger カウンタ由来なので保存しない(load 側で 0 に戻す)。
            # OFF ランでは _assets_state 自体が無い → None=挙動不変(load は .get で旧 ckpt 互換)。
            "assets_state": (
                {"day": sim._assets_state["day"], "prev": sim._assets_state["prev"],
                 "tau": sim._assets_state["tau"]}
                if getattr(sim, "_assets_state", None) else None),
            # 負の評判 第61バッチ c: 種/伝播/忘却の日境界進行(gossip_day)+ 当日カウンタ(gossip_state)+
            # 未ロールの種候補(gossip_pending)。agent 側の _gossip_known/_gossip_heard は agents pickle に
            # 自然同梱。watermark はプロセス内 logger カウンタ由来なので保存しない(load で 0 に戻す=assets
            # と同流儀)。OFF ランでは -1/None=挙動不変(load は .get で旧 checkpoint 互換)。
            "gossip_day": getattr(sim, "_gossip_day", -1),
            "gossip_state": getattr(sim, "_gossip_state", None),
            "gossip_pending": getattr(sim, "_gossip_pending", None),
            # 共同行動 第62バッチ: 日境界進行(joint_day)+当日の成立グループ(logged フラグ込み)+
            # 累積件数(joint_total=L2 n_joint_activity)。従来未保存で mid-day resume が同じ日を
            # 再編成し joint_activity を二重記録し得た(既存ギャップ)。承諾内生化(joint_invite が
            # 日境界で出る)で resume==straight を成立させるため中央管理へ。agent 側 joint_today は
            # agents pickle に自然同梱。endo_state は当日の誘いタリー(承諾/内生判定/履行の set は
            # membership/len のみ使用=集合反復順非保存の影響なし)。OFF ランでは -1/[]/0/None=挙動不変
            # (load は .get で旧 checkpoint 互換)。
            "joint_day": getattr(sim, "_joint_day", -1),
            "joint_groups": getattr(sim, "_joint_groups", []),
            "joint_total": getattr(sim, "_joint_total", 0),
            "endo_state": getattr(sim, "_endo_state", None),
            # 第64バッチ: 誘い経路の当日タリー(invite ON 時のみ存在。int のみ=set なし=
            # 決定論監査は自明)。OFF ランでは None=挙動不変(load は .get で旧 checkpoint 互換)。
            "invite_state": getattr(sim, "_invite_state", None),
            # 第65バッチ: 会話由来 magnitude の当日タリー(quality ON 時のみ存在。int/float のみ)。
            # mid-day checkpoint でも当日平均(L2 quality_magnitude_mean)が resume==straight に
            # なるよう中央管理する。OFF ランでは None=挙動不変(load は .get で旧 ckpt 互換)。
            "quality_state": getattr(sim, "_quality_state", None),
            # 第70バッチ IDEA①(エコー計測): rolling 窓のタリー(deque + カウンタ)。L2 の 5 列は
            # **常設**なので、これを保存しないと mid-day resume の L2 が straight と食い違う
            # (第62の joint 状態と同じ既存ギャップの型)。processed カウンタは**プロセス内 logger
            # 由来**なので保存しない(load 側で 0 に戻す=assets/gossip と同流儀)。集合は使わず
            # deque/dict/Counter のみ=pickle の集合反復順非保存の影響を受けない。
            # observer.echo.enabled=false のランでは state 自体が生えない → None=挙動不変。
            "echo_state": getattr(sim, "_echo_state", None),
            # 第74バッチ IDEA③(規範化ステージ): 語ごとの 4 段階台帳(素の dict/list のみ)。
            # 既定 OFF(labeling.norm_stage.enabled=false)では state 自体が生えない → None=
            # 挙動不変。ON のときは L2 の 2 列が累積量なので、保存しないと mid-day resume の
            # L2 が straight と食い違う(第62 joint / 第70 echo と同じ型のギャップ)。
            # processed カウンタは**プロセス内 logger 由来**なので保存しない(load 側で 0)。
            "norm_state": getattr(sim, "_norm_state", None),
            # 第75バッチ IDEA⑤(ダンバー認知枠): 活性関係数の日次スナップ+休眠/再会の累積カウンタ
            # (int/float のみ)。**休眠状態そのもの**(rel の dormant/dormant_closeness/
            # dormant_step)と直近接触 step(rel の last_step)は agents pickle に自然同梱される
            # ので、ここで中央管理するのは L2 3 列の材料だけ。保存しないと mid-day resume の
            # dormant_total / rekindle_total(累積量)が straight と食い違う(第62 joint /
            # 第70 echo / 第74 norm_state と同じ型のギャップ)。既定 OFF では state 自体が
            # 生えない → None=挙動不変(load は .get で旧 checkpoint 互換)。
            "dunbar_state": getattr(sim, "_dunbar_state", None),
            # 第73バッチ(真偽台帳 Part B): sim 側の台帳=fact レコード(_tl_facts)・重複判定キー
            # (_tl_keys)・累積カウンタ(_tl_stats)。**保存しないと resume 後に fact_id が振り直され、
            # 信念(agents pickle に自然同梱される _fact_beliefs)の参照先が全部迷子になる**
            # (第62 joint / 第70 echo と同型の「日次/累積状態の未保存」ギャップ)。watermark は
            # プロセス内 logger カウンタ由来なので保存しない(load 側で 0 に戻す=assets/gossip 流儀)。
            # beliefs OFF のランでは属性自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            "tl_facts": getattr(sim, "_tl_facts", None),
            "tl_keys": getattr(sim, "_tl_keys", None),
            "tl_stats": getattr(sim, "_tl_stats", None),
            # 第80バッチ W2(検収で顕在化した**既存ギャップ**): 日付・天気の日境界進行
            # (_cal_day)と当日確定値(date_line / weather / weather_line)。従来未保存
            # だったため、weather.enabled=true の mid-day resume は再開直後の 1 step で
            # _phase_calendar_weather が同じ日を再処理し、**weather イベントを二重記録**
            # (+ rain_grievance>0 なら不快感を二重加算)していた=resume≠straight。
            # 第62 joint / 第70 echo / 第75 dunbar と同じ型のギャップ。天気の値そのものは
            # 日インデックスの決定論関数なので保存しない(復元は当日値のキャッシュのみ)。
            # calendar/weather OFF のランでは -1/None = 挙動不変(load は .get で旧 ckpt 互換)。
            "cal_day": getattr(sim, "_cal_day", -1),
            "today_date_line": getattr(sim, "today_date_line", None),
            "today_weather": getattr(sim, "today_weather", None),
            "today_weather_line": getattr(sim, "today_weather_line", None),
            # 第71バッチ: LLM 入出力ジャーナルの確定点(ファイル名 → {records, bytes})。
            # mark() が flush してから採るので、この時点のファイル末尾は必ず gzip メンバ境界
            # = 安全な切り詰め点。resume(load)がここまで巻き戻すことで、「checkpoint 後に
            # 走ってクラッシュした分」を再走したときの二重記録と seq の巻き戻りを両方防ぐ。
            # journal OFF のランでは {} = 挙動不変(load は .get で旧 checkpoint 互換)。
            "llm_journal": (sim._journal_marks()
                            if hasattr(sim, "_journal_marks") else {}),
        },
        # --- scenario は config から再構築される。封鎖の進行だけを直列化(順序安定) ---
        "scenario": {
            "active": bool(sim.scenario.active),
            "closed": sorted(sim.scenario.closed),   # set[tuple] → sorted list
        },
        # --- traffic は city/router/hub 付きで再構築される。可変ランタイムのみ ---
        "traffic": {
            "cars": sim.traffic.cars,
            "n_active": sim.traffic.n_active,
            "last_n": sim.traffic.last_n,
            "total_spawned": sim.traffic.total_spawned,
        },
    }
    raw = pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)
    return path


def load(sim, path: str | Path) -> int:
    """checkpoint を sim へ上書き復元し、次に実行すべき step を返す。"""
    with gzip.open(Path(path), "rb") as f:
        blob = pickle.loads(f.read())
    if blob.get("format") != FORMAT:
        raise ValueError(f"未知の checkpoint format: {blob.get('format')}")
    expect = config_hash(sim.cfg)
    if blob.get("config_hash") != expect:
        raise ValueError(
            "checkpoint の config が現在の config と不整合(決定論が壊れる)。"
            " seed/n_agents/因子など resume 対象外のキーが変わっている可能性。")

    # 状態オブジェクト(共有参照は 1 pickle 内で保存済み)
    sim.agents = blob["agents"]
    sim.agent_by_id = {a.id: a for a in sim.agents}
    sim.labels = blob["labels"]
    sim.items = sim.labels.items            # ItemStore を labels と同一実体に固定
    sim.net = blob["net"]
    sim.tools = blob["tools"]
    sim.rulebook = blob["rulebook"]         # 制度DSL: active ルールを跨いで復元
    sim.world_events = blob["world_events"]

    rt = blob["runtime"]
    sim.drive_stats = rt["drive_stats"]
    sim._econ_day = rt["econ_day"]
    sim._acct_day = rt.get("acct_day", -1)      # 旧 checkpoint 互換(無ければ -1)
    sim._org_day = rt.get("org_day", {})        # B4: 会社観測 per-day アキュムレータ(旧 checkpoint 互換)
    sim._org_ledger_day = rt.get("org_ledger_day", -1)
    sim._housing_day = rt.get("housing_day", -1)  # 第60バッチ b: 転居/同棲の日境界進行(旧 checkpoint 互換)
    sim._career_day = rt.get("career_day", -1)    # 第60バッチ b: career 日境界進行(旧 checkpoint 互換)
    sim._rel_day = rt.get("rel_day", -1)          # 第61検収補修: 社会関係 日境界進行(旧 checkpoint 互換)
    ast = rt.get("assets_state")                # 第59: 資産レンズ τ の前日状態(旧 checkpoint 互換=無ければ素通り)
    if ast:
        sim._assets_state = {"day": ast["day"], "prev": ast["prev"],
                             "tau": ast["tau"], "processed": 0}
    sim._gossip_day = rt.get("gossip_day", -1)  # 第61 c: 負の評判の日境界進行(旧 checkpoint 互換)
    gst = rt.get("gossip_state")
    if gst is not None:
        sim._gossip_state = gst
    gpd = rt.get("gossip_pending")
    if gpd is not None:
        sim._gossip_pending = gpd
    sim._gossip_watermark = 0                   # watermark は load で 0(fresh logger=total 0 から再走査。assets と同流儀)
    # 共同行動 第62バッチ: 日境界進行+当日グループ+累積件数+承諾内生化タリー(旧 checkpoint 互換=
    # 無ければ従来どおり -1/[]/0 で再編成へ後退)。
    sim._joint_day = rt.get("joint_day", -1)
    jg = rt.get("joint_groups")
    if jg is not None:
        sim._joint_groups = jg
    sim._joint_total = rt.get("joint_total", 0)
    est = rt.get("endo_state")
    if est is not None:
        sim._endo_state = est
    ivs = rt.get("invite_state")                # 第64: 誘い経路タリー(旧 checkpoint 互換=無ければ素通り)
    if ivs is not None:
        sim._invite_state = ivs
    qst = rt.get("quality_state")               # 第65: 会話の質タリー(旧 checkpoint 互換=無ければ素通り)
    if qst is not None:
        sim._quality_state = qst
    est_echo = rt.get("echo_state")             # 第70 IDEA①: エコー窓(旧 checkpoint 互換=無ければ素通り)
    if est_echo is not None:
        sim._echo_state = est_echo
    sim._echo_processed = 0                     # fresh logger=total 0 から再走査(assets と同流儀)
    sim._echo_cache = None
    nst = rt.get("norm_state")                  # 第74 IDEA③: 規範化ステージ台帳(旧 ckpt 互換=無ければ素通り)
    if nst is not None:
        sim._norm_state = nst
    sim._norm_processed = 0                     # fresh logger=total 0 から再走査(echo と同流儀)
    sim._norm_cache = None
    dst = rt.get("dunbar_state")                # 第75 IDEA⑤: 認知枠タリー(旧 ckpt 互換=無ければ素通り)
    if dst is not None:
        sim._dunbar_state = dst
    # 第73 Part B: 真偽台帳(旧 checkpoint 互換=無ければ素通り=beliefs OFF の挙動不変)。
    # canary は module 側のプロセス内レジストリなので、復元した fact 分を再登録して武装を保つ。
    tlf = rt.get("tl_facts")
    if tlf is not None:
        from ..truth_ledger import register_canary as _tl_canary
        sim._tl_facts = tlf
        for _fid in sorted(tlf):
            _tl_canary(_fid)
    tlk = rt.get("tl_keys")
    if tlk is not None:
        sim._tl_keys = tlk
    tls = rt.get("tl_stats")
    if tls is not None:
        sim._tl_stats = tls
    sim._tl_watermark = 0                       # fresh logger=total 0 から再走査(gossip と同流儀)
    # 第80バッチ W2: 日付・天気の日境界進行と当日確定値(旧 checkpoint 互換=無ければ従来どおり
    # -1/None=再開直後に当日を作り直す)。生成モードの日別系列は保存しない — 再構築が
    # (master_seed, params, 起点の日) の純関数で、生成は逐次=prefix 安定だから同一系列になる。
    sim._cal_day = rt.get("cal_day", -1)
    if "today_weather" in rt:
        sim.today_date_line = rt.get("today_date_line")
        sim.today_weather = rt.get("today_weather")
        sim.today_weather_line = rt.get("today_weather_line")
    if hasattr(sim, "_journal_rewind"):         # 第71: LLM ジャーナルを確定点まで巻き戻す
        sim._journal_rewind(rt.get("llm_journal"))

    # scenario: __init__ で config から再構築済み。封鎖の進行を復元し city へ再適用。
    sc = blob["scenario"]
    sim.scenario.active = bool(sc["active"])
    sim.scenario.closed = {tuple(e) for e in sc["closed"]}
    for u, v in sim.scenario.closed:
        if sim.city.graph.has_edge(u, v):
            sim.city.graph.edges[u, v]["closed"] = True
    sim.router.invalidate()                 # 封鎖(または非封鎖)を経路計算へ反映

    # traffic: __init__ で city/router/hub 付きで再構築済み。可変ランタイムだけ戻す。
    tr = blob["traffic"]
    sim.traffic.cars = tr["cars"]
    sim.traffic.n_active = tr["n_active"]
    sim.traffic.last_n = tr["last_n"]
    sim.traffic.total_spawned = tr["total_spawned"]

    return int(blob["step"])


def latest(run_dir: str | Path) -> Path | None:
    """run_dir/checkpoint/ 内の最新(step 最大)の checkpoint パス。無ければ None。"""
    ckpt_dir = Path(run_dir) / "checkpoint"
    if not ckpt_dir.is_dir():
        return None
    cands = sorted(ckpt_dir.glob("ckpt-*.pkl.gz"))
    if not cands:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name[len("ckpt-"):].split(".")[0])
        except ValueError:
            return -1

    return max(cands, key=_step)
