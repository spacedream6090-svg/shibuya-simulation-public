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


def config_hash(cfg) -> str:
    """決定論に効く config の内容ハッシュ(resume 制御キーは除外)。"""
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(cfg, resolve=True)
    data = copy.deepcopy(data)
    for sect, key in _VOLATILE_KEYS:
        if isinstance(data.get(sect), dict):
            data[sect].pop(key, None)
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
