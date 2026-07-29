"""環境改変条件(`world.mod`)= 現実幾何を原点とした反実仮想のパラメータ化(A1 第67バッチ)。

`world/scenario.py` の `shock_closure`(半径内の道路封鎖を or 時限で発動)を **一般化** した層。
scenario が「実行中に起きる摂動(時間軸あり)」なのに対し、こちらは **ワールド構築時に一度だけ・
決定論・乱数ゼロで適用される静的な条件**(=その世界が最初からそうであった、という反実仮想)。
現実の街の幾何が原点(`enabled: false` = 空 = 何も改変しない)で、条件はその周りの近傍として置く。

改変の初期セット(4種):
  1. `edges_closed`      … 指定エッジ集合を無効化する(既存封鎖の一般化。routing が迂回する)
  2. `edge_speed_scale`  … 指定エッジ集合の速度係数(歩道幅の代理)。走行コスト長へ写して
                            経路選択と移動所要 step の双方に効かせる(CityMap.scale_edge_cost)
  3. `open_hours`        … 営業時間のオーバーライド(cat 単位。POI 単位は予約=後述)
  4. `gate_capacity`     … 駅ゲート容量の係数(**予約フィールド・本バッチでは未消費**=後述)

エッジ集合の指定は 2 通り(どちらも同一セレクタ構造):
  - `edges`: `[[u, v], ...]` の明示リスト(順不同。`{u: .., v: ..}` 形式も受ける)
  - `bbox` : `[[x0, y0, x1, y1], ...]` の矩形。**端点のいずれかが矩形内**なら選択
             (`scenario._edges_in_radius` の「端点いずれかが半径内」と同じ規則の矩形版)
両方書いたら和集合。地図ローカル座標 [m]。

R1 契約(重要): この層は `k.writeback` / beliefs / 内部の構成概念を **一切読まない**。参照するのは
  conf(プロファイル)と地図の幾何(ノード座標・エッジ)だけである。LLM 呼び出しは 1 本も足さない。
  改変が LLM 呼数へ及ぼす影響は「誰がどこに居合わせるか(co-location)が変わる」物理経由のみで、
  これは計画書 §5-1 が明示的に許容する範囲(k 不変性は controls.mode=compute_matched +
  test_*_call_count_k_invariant で担保する)。

no-fingerprint 契約: このファイルは `world/` 配下=静的検査の対象。地名・業種名・構成概念名の
  リテラルを一切書かない。cat 名も座標もノード ID も **すべてプロファイル(conf 側)から来る**。

既定 OFF(`world.mod.enabled: false`): `load()` が即 None を返し、**プロファイルファイルを開きも
  しない**。地図・エッジ属性・commerce 設定のいずれも触らない=新イベント 0 件・新状態なし・
  L1 バイト一致・draw 数同一(乱数は元々使わない)。

★ `gate_capacity` について正直な但し書き: 現行エンジンに「駅ゲートの容量」という概念は存在しない
  (`world.edge_capacity` は道路エッジの混雑減速であって改札ではない)。したがって本バッチでは
  **スキーマ上のフィールドとして受理・検証・summary への記録までを行い、実効は持たせない**。
  値を消費するのは将来バッチ(計画書レーン3 の H_B / 屋外物理層)。
★ `scenario.shock_closure` との併用は本バッチでは想定していない: 両者は同じ `edge["closed"]`
  フラグを使い、shock_closure の解除は「自分が閉じた集合」を無条件に pop するため、静的条件で
  閉じたエッジまで開いてしまう。併用が要るようになったら scenario 側の解除を「自分が立てた分
  だけ」に限定する改修を先に入れること(現状は world.mod と scenario を同時に使わない運用)。
★ `open_hours.pois`(POI 単位のオーバーライド)も同様に **予約フィールド**。現行の
  `commerce.is_open_poi` は POI の cat しか見ないため、POI 単位を効かせるには commerce 層の
  改修が要る(本バッチの変更範囲外)。cat 単位(`open_hours.cats`)は実効する。
"""
from __future__ import annotations

from pathlib import Path

# --- 受理するプロファイルのトップレベルキー(未知キーは明確なエラーにする=実験条件の取り違え防止)
_PROFILE_KEYS = {"name", "description", "edges_closed", "edge_speed_scale",
                 "open_hours", "gate_capacity"}
_SELECTOR_KEYS = {"edges", "bbox"}


def edge_key(u: str, v: str) -> tuple[str, str]:
    """無向エッジの正準キー(端点を昇順に並べる)。scenario と共有する唯一の同一性規則。"""
    return (u, v) if u <= v else (v, u)


def _to_plain(obj):
    """OmegaConf の DictConfig/ListConfig を素の dict/list へ(スカラーはそのまま)。"""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:                                  # pragma: no cover - omegaconf 不在環境
        pass
    return obj


def _pair(item) -> tuple[str, str]:
    """明示エッジ 1 件 → (u, v)。`[u, v]` / `{"u": .., "v": ..}` のどちらでも受ける。"""
    if isinstance(item, dict):
        return str(item["u"]), str(item["v"])
    seq = list(item)
    if len(seq) != 2:
        raise ValueError(f"world.mod: エッジ指定は [u, v] の 2 要素であること: {item!r}")
    return str(seq[0]), str(seq[1])


def _bbox(item) -> tuple[float, float, float, float]:
    seq = [float(c) for c in list(item)]
    if len(seq) != 4:
        raise ValueError(f"world.mod: bbox は [x0, y0, x1, y1] の 4 要素であること: {item!r}")
    x0, y0, x1, y1 = seq
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def select_edges(city, selector: dict) -> list[tuple[str, str]]:
    """セレクタ(edges / bbox)→ 正準キーのエッジ一覧(昇順=決定論・重複なし)。

    - `edges`: 明示 (u,v)。地図に無いエッジは黙って捨てず ValueError(条件の取り違え防止)。
    - `bbox` : 端点のいずれかが矩形内(境界含む)なら選択。
    どちらも無い/空なら空リスト(=何も選ばない)。乱数を一切引かない純関数。
    """
    selector = dict(_to_plain(selector) or {})
    # edge_speed_scale のルールはセレクタと係数を 1 つの dict に書くので "scale" も通す
    # (係数の解釈は呼び出し側 apply_world の責務。ここでは選択だけを行う)。
    unknown = set(selector) - _SELECTOR_KEYS - {"scale"}
    if unknown:
        raise ValueError(f"world.mod: 未知のセレクタキー {sorted(unknown)} "
                         f"(受理するのは {sorted(_SELECTOR_KEYS)} + 'scale')")
    out: set[tuple[str, str]] = set()
    for item in list(selector.get("edges") or []):
        u, v = _pair(item)
        if not city.graph.has_edge(u, v):
            raise ValueError(f"world.mod: 地図に存在しないエッジ ({u}, {v}) が指定された")
        out.add(edge_key(u, v))
    boxes = [_bbox(b) for b in list(selector.get("bbox") or [])]
    if boxes:
        for u, v in city.graph.edges():
            ux, uy = city.node_xy(u)
            vx, vy = city.node_xy(v)
            for x0, y0, x1, y1 in boxes:
                if ((x0 <= ux <= x1 and y0 <= uy <= y1)
                        or (x0 <= vx <= x1 and y0 <= vy <= y1)):
                    out.add(edge_key(u, v))
                    break
    return sorted(out)


class WorldMod:
    """1 本の環境改変条件(名前つきプロファイル)。適用はワールド構築時の一度きり。"""

    def __init__(self, name: str, doc: dict, path: Path | None = None):
        self.name = str(name)
        self.path = path
        doc = dict(_to_plain(doc) or {})
        unknown = set(doc) - _PROFILE_KEYS
        if unknown:
            raise ValueError(f"world.mod: プロファイル '{name}' に未知のキー {sorted(unknown)} "
                             f"(受理するのは {sorted(_PROFILE_KEYS)})")
        self.doc = doc
        self.description = str(doc.get("description", "") or "")
        # 適用実績(summary.json 用。適用前は空)
        self.applied: dict = {}
        self.reserved: dict = {}

    # ------------------------------------------------------------ 世界(地図)への適用
    def apply_world(self, city) -> dict:
        """地図へ改変を適用する(決定論・乱数ゼロ・1 回だけ)。戻り値=適用実績。

        呼び出し側は適用後に `router.invalidate()` して経路キャッシュを捨てること
        (`scenario.shock_closure` と同じ規約)。
        """
        # 1) エッジ無効化(既存 shock_closure と同じ closed フラグ=routing が除外する)
        closed = select_edges(city, self.doc.get("edges_closed") or {})
        for u, v in closed:
            city.graph.edges[u, v]["closed"] = True
        self.applied["edges_closed"] = {"n_edges": len(closed)}

        # 2) 速度係数(実効長へ写す。同一エッジが複数ルールに該当したら記述順で後勝ち)
        rules = list(_to_plain(self.doc.get("edge_speed_scale")) or [])
        applied_rules = []
        for i, rule in enumerate(rules):
            rule = dict(_to_plain(rule) or {})
            if "scale" not in rule:
                raise ValueError(f"world.mod: edge_speed_scale[{i}] に scale が無い")
            scale = float(rule["scale"])
            if scale <= 0.0:
                raise ValueError(f"world.mod: edge_speed_scale[{i}].scale は正の数であること: {scale}")
            picked = select_edges(city, rule)
            n_ok = sum(1 for u, v in picked if city.scale_edge_cost(u, v, scale))
            applied_rules.append({"scale": scale, "n_selected": len(picked),
                                  "n_applied": n_ok})
        if applied_rules:
            self.applied["edge_speed_scale"] = applied_rules

        # 3) 駅ゲート容量 = 予約フィールド(現行エンジンに容量概念が無い。冒頭 docstring 参照)。
        #    値の検証と記録だけ行い、世界には一切触れない(=未消費であることを summary で明示)。
        gate = dict(_to_plain(self.doc.get("gate_capacity")) or {})
        if gate:
            for k, v in gate.items():
                if float(v) <= 0.0:
                    raise ValueError(f"world.mod: gate_capacity['{k}'] は正の数であること: {v}")
            self.reserved["gate_capacity"] = {"n_entries": len(gate), "consumed": False}
        return self.applied

    # ------------------------------------------------------------ commerce への適用
    def apply_commerce(self, commercecfg: dict | None) -> dict:
        """営業時間オーバーライド(cat 単位)を commerce 設定へ書き込む(決定論・乱数ゼロ)。

        `commercecfg["hours"]` は cat -> (open_min, close_min)。プロファイルは時(24h 表記)の
        `[open, close]` で書き、ここで分へ変換する(`commerce.build_cfg` と同じ規約)。
        commerce.enabled=false のときは `filter_open` がそもそもゲートしないので**実効はない**
        (設定だけ書き換わる)。その旨は summary の `commerce_enabled` で分かるようにする。
        """
        oh = dict(_to_plain(self.doc.get("open_hours")) or {})
        unknown = set(oh) - {"cats", "pois"}
        if unknown:
            raise ValueError(f"world.mod: open_hours の未知キー {sorted(unknown)} "
                             "(受理するのは ['cats', 'pois'])")
        cats = dict(_to_plain(oh.get("cats")) or {})
        applied: dict[str, list[int]] = {}
        if cats and commercecfg is not None:
            for cat in sorted(cats):
                hc = list(cats[cat])
                if len(hc) != 2:
                    raise ValueError(f"world.mod: open_hours.cats['{cat}'] は [開店時, 閉店時]")
                o, c = int(hc[0]), int(hc[1])
                commercecfg["hours"][str(cat)] = (o * 60, c * 60)
                applied[str(cat)] = [o, c]
        if applied:
            self.applied["open_hours_cats"] = applied
            self.applied["commerce_enabled"] = bool(commercecfg
                                                    and commercecfg.get("enabled"))
        # POI 単位は予約フィールド(冒頭 docstring 参照)。検証と記録のみ=未消費。
        pois = dict(_to_plain(oh.get("pois")) or {})
        if pois:
            for pid in sorted(pois):
                hc = list(pois[pid])
                if len(hc) != 2:
                    raise ValueError(f"world.mod: open_hours.pois['{pid}'] は [開店時, 閉店時]")
            self.reserved["open_hours_pois"] = {"n_entries": len(pois), "consumed": False}
        return self.applied

    # ------------------------------------------------------------ 記録
    def summary(self) -> dict:
        """summary.json の `world_mod` キーへ入れる内容(profile 名 + 適用実績 + 予約フィールド)。"""
        out: dict = {"profile": self.name, "applied": dict(self.applied)}
        if self.description:
            out["description"] = self.description
        if self.path is not None:
            # リポジトリ相対で残す(ラン間で比較可能・ローカル絶対パスを summary に混ぜない)
            root = Path(__file__).resolve().parents[3]
            p = Path(self.path)
            out["path"] = str(p.relative_to(root)).replace("\\", "/") \
                if p.is_absolute() and p.is_relative_to(root) else str(p)
        if self.reserved:
            # 「スキーマは受理したが世界には効かせていない」ことを機械可読に残す(正直な記録)
            out["reserved_not_consumed"] = dict(self.reserved)
        return out


def resolve_profile_path(profile: str, repo_root: Path) -> Path:
    """プロファイル指定 → ファイルパス。裸の名前は conf/worldmod/<name>.yaml として解決する。"""
    s = str(profile)
    p = Path(s)
    if p.suffix in (".yaml", ".yml") or len(p.parts) > 1:
        return p if p.is_absolute() else (repo_root / p)
    return repo_root / "conf" / "worldmod" / f"{s}.yaml"


def load(cfg, repo_root: Path) -> WorldMod | None:
    """conf の `world.mod` を読み、有効なら WorldMod を返す。既定 OFF なら **None**。

    OFF のときはプロファイルファイルを開かない(I/O ゼロ)=世界に何も起きない。
    enabled=true なのに profile が空、あるいはファイルが無い場合は明確なエラー
    (実験条件の取り違えを黙って通さない)。
    """
    world = cfg.get("world", {}) or {}
    mcfg = world.get("mod", None)
    if mcfg is None or not bool(mcfg.get("enabled", False)):
        return None
    profile = mcfg.get("profile", None)
    if not profile:
        raise ValueError("world.mod.enabled=true だが world.mod.profile が未指定"
                         "(conf/worldmod/<name>.yaml の名前を指定する)")
    path = resolve_profile_path(str(profile), Path(repo_root))
    if not path.exists():
        raise FileNotFoundError(f"world.mod のプロファイルが見つからない: {path}")
    from omegaconf import OmegaConf
    doc = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    name = str(doc.get("name") or Path(str(profile)).stem)
    return WorldMod(name, doc, path=path)
