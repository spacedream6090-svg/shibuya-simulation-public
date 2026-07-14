#!/usr/bin/env python
"""創発の「後付けテキスト検出」(第P1バッチ・読み出し専用)。

    python scripts/detect_emergence.py                 # l1_events.parquet を持つ最新ラン
    python scripts/detect_emergence.py --run wv_llm_7d  # ラン名を指定

旧実装 shibuya-sim/scripts/detect_emergence.py の思想を現行リポジトリ(parquet L1)へ
移植した**分析専用**スクリプト。runs/<name> の l1_events.parquet(payload は JSON 文字列)・
config.yaml(world.map の地図データ)・agents.json だけを読み、sim 本体・schema は一切
変更しない(src/ は読まない=完全に後処理)。

検出器(いずれも保守的なヒューリスティクス。命中は「観測候補」であって断定ではない):
  1. fiction   : テキスト中の固有名詞候補(カタカナ連続・「」引用・ASCII 固有名詞)を
                 実在名ホワイトリスト(地図 POI/建物/駅/路線・エージェント名・正規造語)と
                 照合し、未知(=架空 or 一般語)を文脈つきで列挙。
  2. norms     : 規範発話の日本語/英語パターン(〜べき・してはいけない・普通/常識・
                 ものだ・マナー・ルールを守る・みんな〜する 等)を1発話1件で検出。
  3. attention : 同一語が窓 24step(=4時間)以内に 3人以上の別エージェントで共伝播した語を
                 抽出(ストップワード・ホワイトリスト名は除外)。造語は provenance="coined"、
                 それ以外は "organic"。

出力: runs/<run>/emergence.json(全件)・runs/<run>/emergence_report.md(日本語・見どころ)。
依存は標準ライブラリ + pyarrow + omegaconf(いずれも既存依存)のみ。ステップ=10分・1日=144step。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# Windows コンソール(cp932)対策。ファイル出力は常に UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RUNS_ROOT = os.path.join(_ROOT, "runs")

STEPS_PER_DAY = 144
WINDOW_STEPS = 24        # 4時間(=24×10分)
MIN_ATTENTION_AGENTS = 3

# テキストを持つイベント種と、その本文フィールド(schema.py 準拠)
#   speak/sns_post/dm は payload["text"]、reflect は written_back=True の belief のみ。
_TEXT_KINDS = ("speak", "sns_post", "dm", "reflect")


# --------------------------------------------------------------------------- #
# ローダ(build_panel.py / measure.load_events と同じ流儀。ここでは自前で最小実装し
# numpy 等を引き込まない。返り値は {step, agent, kind, payload(dict)})
# --------------------------------------------------------------------------- #
def load_events(run_dir: str) -> list[dict]:
    """l1_events.parquet を列射影 + RecordBatch 逐次で読み、payload を dict 展開して返す。"""
    import pyarrow.parquet as pq

    path = os.path.join(run_dir, "l1_events.parquet")
    want = ["step", "sim_min", "agent_id", "kind", "payload"]
    available = set(pq.read_schema(path).names)
    cols = [c for c in want if c in available]
    pf = pq.ParquetFile(path)
    out: list[dict] = []
    for batch in pf.iter_batches(columns=cols):
        d = batch.to_pydict()
        n = len(d["step"])
        step, agent_id, kind, pays = d["step"], d["agent_id"], d["kind"], d["payload"]
        for i in range(n):
            raw = pays[i]
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append({
                "step": int(step[i]),
                "agent": int(agent_id[i]),
                "kind": kind[i],
                "payload": payload,
            })
    return out


def load_agents(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "agents.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_map_for_run(run_dir: str) -> dict | None:
    """ランの config.yaml が参照する world.map(地図 JSON)を読み込む。無ければ None。"""
    cfg_path = os.path.join(run_dir, "config.yaml")
    map_rel = None
    if os.path.exists(cfg_path):
        try:                                              # 既存依存 omegaconf を優先
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(cfg_path)
            map_rel = cfg.get("world", {}).get("map")
        except Exception:
            map_rel = _grep_map_path(cfg_path)            # 最小フォールバック
    if not map_rel:
        return None
    map_path = map_rel if os.path.isabs(map_rel) else os.path.join(_ROOT, map_rel)
    if not os.path.exists(map_path):
        return None
    with open(map_path, encoding="utf-8") as fh:
        return json.load(fh)


def _grep_map_path(cfg_path: str) -> str | None:
    """yaml パーサ無しでも `  map: data/xxx.json` 行だけは拾うフォールバック。"""
    pat = re.compile(r"^\s*map:\s*([^\s#]+)")
    with open(cfg_path, encoding="utf-8") as fh:
        for line in fh:
            mm = pat.match(line)
            if mm:
                return mm.group(1).strip().strip('"\'')
    return None


def _load_config(run_dir: str) -> dict | None:
    """ランの config.yaml を dict 相当で返す(omegaconf 優先・無ければ None)。"""
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(cfg_path)
    except Exception:
        return None


def load_orgs_for_run(run_dir: str) -> dict | None:
    """ランの config.yaml が参照する organizations.book(組織台帳 JSON)を読む。無ければ None。

    組織台帳は build_orgs.py が生成する架空の会社/学校の名簿(R17 で実在企業名は入らないが、
    シミュ環境の「公式な実在物」なので接地判定では実在名として扱う)。"""
    cfg = _load_config(run_dir)
    book_rel = None
    if cfg is not None:
        try:
            book_rel = cfg.get("organizations", {}).get("book")
        except Exception:
            book_rel = None
    if not book_rel:                                     # 最小フォールバック(grep)
        cfg_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(cfg_path):
            pat = re.compile(r"^\s*book:\s*([^\s#]+)")
            with open(cfg_path, encoding="utf-8") as fh:
                for line in fh:
                    mm = pat.match(line)
                    if mm:
                        book_rel = mm.group(1).strip().strip('"\'')
                        break
    if not book_rel:
        return None
    book_path = book_rel if os.path.isabs(book_rel) else os.path.join(_ROOT, book_rel)
    if not os.path.exists(book_path):
        return None
    with open(book_path, encoding="utf-8") as fh:
        return json.load(fh)


def load_model_backend(run_dir: str) -> str | None:
    """ランの config.yaml から model.backend(ollama / mock / …)を返す。判定できなければ None。

    mock ラン(テンプレ由来の発話)は接地率が無意味に高く出るため、レポートで注記する用途。"""
    cfg = _load_config(run_dir)
    if cfg is not None:
        try:
            be = cfg.get("model", {}).get("backend")
            if be:
                return str(be)
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
# ホワイトリスト(実在名)+ 正規造語
# --------------------------------------------------------------------------- #
def whitelist_names(map_data: dict | None, agents: list[dict],
                    orgs: dict | None = None) -> list[str]:
    """実在名リストを作る = 地図(POI/建物/駅/路線/命名ノード)+ エージェント名簿
    (氏名・勤務先・沿線)+ 組織台帳(会社・学校)。orgs は後方互換のため任意。"""
    names: set[str] = set()
    if map_data:
        for poi in map_data.get("pois", []) or []:
            nm = (poi.get("name") or "").strip()
            if nm:
                names.add(nm)
        for b in map_data.get("buildings", []) or []:
            nm = (b.get("name") or "").strip()
            if nm:
                names.add(nm)
        for r in map_data.get("railways", []) or []:
            nm = (r.get("name") or "").strip()
            if nm:
                names.add(nm)
        for nd in map_data.get("nodes", []) or []:
            nm = (nd.get("name") or "").strip() if isinstance(nd, dict) else ""
            if nm:
                names.add(nm)
    for a in agents or []:
        for key in ("name", "work_name", "part_time_name", "residence_line"):
            nm = (a.get(key) or "").strip() if isinstance(a.get(key), str) else ""
            if nm:
                names.add(nm)
    if orgs:                                             # 組織台帳(会社+学校の名称)
        for sec in ("companies", "schools"):
            for o in orgs.get(sec, []) or []:
                nm = (o.get("name") or "").strip() if isinstance(o, dict) else ""
                if nm:
                    names.add(nm)
    return sorted(names)


def collect_coined(events: list[dict]) -> list[dict]:
    """シミュ内で正規に造語された語(label_coin / vocab_coin)を集める。

    返り値: [{text, item_id, step, agent}]。造語は「架空」ではなく既観測なので
    fiction からは除外し、別カテゴリ "coined" として集計する。
    """
    out: list[dict] = []
    for e in events:
        if e["kind"] in ("label_coin", "vocab_coin"):
            txt = (e["payload"].get("text") or "").strip()
            if txt:
                out.append({
                    "text": txt,
                    "item_id": e["payload"].get("item_id"),
                    "step": e["step"],
                    "agent": e["agent"],
                })
    return out


# 「シミュ内で正当に創発した固有名(=シミュ内実在)」を持つイベント種と、名前フィールド。
#   造語(label_coin/vocab_coin)は text、それ以外(出店・コミュニティ・制度)は name。
#   これらは「架空」ではなく sim 内で実体を持つので、作話(fiction)と混同しない。
_SIM_NAME_KINDS = {
    "label_coin": "text", "vocab_coin": "text",            # 新語・ラベル
    "venture_open": "name", "venture_close": "name",       # 出店の店名
    "venture_permit": "name", "venture_fulltime": "name",
    "group_found": "name", "group_join": "name",           # コミュニティ名
    "institution": "name", "institution_rule": "name",     # 制度・status function 名
}


def collect_sim_names(events: list[dict]) -> list[dict]:
    """シミュ内で正当に創発した固有名(造語・店名・コミュニティ名・制度名)を集める。

    返り値: [{text, kind, step, agent}]。接地判定で「シミュ内実在(=創発名)」カテゴリの
    照合台帳に使う。同名の重複は保持(初出 step の把握は呼び出し側で行う)。"""
    out: list[dict] = []
    for e in events:
        field = _SIM_NAME_KINDS.get(e["kind"])
        if not field:
            continue
        nm = (e["payload"].get(field) or "").strip()
        if nm:
            out.append({"text": nm, "kind": e["kind"],
                        "step": e["step"], "agent": e["agent"]})
    return out


# --------------------------------------------------------------------------- #
# テキスト抽出(生イベント -> 検出器が食べる純データ)
# --------------------------------------------------------------------------- #
def extract_text_records(events: list[dict]) -> list[dict]:
    """テキストを持つイベントを {step, agent, kind, text} の列に正規化して返す。

    speak/sns_post/dm は payload["text"]。reflect は written_back=True の belief のみ
    (内省の書き戻し=当人の定着した信念のみを対象にする)。agent_id<0(世界イベント)は除く。
    """
    recs: list[dict] = []
    for e in events:
        kind, aid, p = e["kind"], e["agent"], e["payload"]
        if aid is None or aid < 0:
            continue
        if kind in ("speak", "sns_post", "dm"):
            text = (p.get("text") or "").strip()
        elif kind == "reflect":
            if not p.get("written_back"):
                continue
            text = (p.get("belief") or "").strip()
        else:
            continue
        if text:
            recs.append({"step": e["step"], "agent": aid, "kind": kind, "text": text})
    return recs


# --------------------------------------------------------------------------- #
# 1. fiction(架空固有名詞)
# --------------------------------------------------------------------------- #
# カタカナ連続(3文字以上)・「」引用・ASCII 固有名詞候補。
_KATAKANA_RE = re.compile(r"[゠-ヿｦ-ﾟ]{3,}")
_QUOTED_RE = re.compile(r"[「『]([^」』]{2,40})[」』]")
_ASCII_PROPER_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'&.\-]{2,}(?:\s+[A-Z][A-Za-z0-9'&.\-]{2,})*\b")


# 汎用のカタカナ一般名詞(=固有名詞ではない普通名詞)。架空名の signal を埋もれさせる
# ノイズなので fiction から除外する(「実在」の主張ではなく「一般語」としての除外)。
_GENERIC_KATAKANA = {
    "データ", "スムージー", "メニュー", "リズム", "ランニング", "アート", "プロジェクト",
    "メロディー", "アイデア", "グラデーション", "テーマ", "ニュース", "スカーフ", "スタート",
    "フレーズ", "メッセージ", "サウナ", "カーブ", "モチーフ", "タイミング", "インスピレーション",
    "コミュニティ", "センサー", "ステップ", "ページ", "チェック", "ツール", "バージョン",
    "スムーズ", "セールス", "バタフライ", "コーヒー", "カフェ", "イベント", "ライブ", "ビル",
    "ボード", "ゲーム", "ランチ", "ディナー", "スイーツ", "ドリンク", "テラス", "ベンチ",
    "エネルギー", "パワー", "スピード", "ストレス", "リラックス", "コンセプト", "デザイン",
    "スタイル", "カラー", "ライト", "サイン", "ポイント", "ルート", "コース", "プラン",
    "スケジュール", "パーティー", "メンバー", "グループ", "チーム", "パートナー", "サポート",
    "エスカレーター", "エレベーター", "スマホ", "スマートフォン", "メートル", "ビール",
    "アプリ", "バッテリー", "ケース", "スケール", "ドアノブ", "キャッチ", "リハーサル",
    "スクリーン", "カメラ", "マイク", "スピーカー", "テーブル", "ソファ", "ドア",
}
# 会話の冒頭・相槌に頻出する英語のありふれた語(ASCII 固有名詞の誤検出)。
_GENERIC_ASCII = {
    "i'm", "hey", "maybe", "ok", "okay", "yeah", "hi", "hello", "yes", "no",
    "wow", "let's", "the", "and", "but", "you", "well", "oh", "hmm",
}


def _extract_candidates(text: str) -> list[tuple[str, str]]:
    """(候補文字列, 種別) の重複なしリスト。種別: katakana / quoted / ascii。"""
    seen: dict[str, str] = {}
    for c in _KATAKANA_RE.findall(text):
        seen.setdefault(c, "katakana")
    for c in _QUOTED_RE.findall(text):
        c = c.strip()
        if len(c) >= 2:
            seen.setdefault(c, "quoted")
    for c in _ASCII_PROPER_RE.findall(text):
        c = c.strip()
        if len(c) >= 3:
            seen.setdefault(c, "ascii")
    return list(seen.items())


class NameIndex:
    """実在名との照合器。exact -> 部分一致 -> SequenceMatcher>=threshold の順に判定。"""

    def __init__(self, names: list[str], threshold: float = 0.7):
        self.threshold = threshold
        self.names = [n for n in names if n]
        self.exact = set(self.names)
        # 「候補が実在名の部分文字列」を高速判定するための連結ブロブ。
        self._blob = "\n".join(self.names)
        # 「実在名が候補の部分文字列」判定を軽くするため、短い名前だけ別持ち。
        self._short = [n for n in self.names if len(n) <= 8]
        self._memo: dict[str, tuple[bool, str | None]] = {}

    def classify(self, cand: str) -> tuple[bool, str | None]:
        """(known?, nearest_name) を返す。known=True は実在名として除外すべき候補。"""
        if cand in self._memo:
            return self._memo[cand]
        res = self._classify(cand)
        self._memo[cand] = res
        return res

    def _classify(self, cand: str) -> tuple[bool, str | None]:
        if cand in self.exact:
            return True, cand
        if cand in self._blob:                     # 候補 ⊂ どれかの実在名
            return True, None
        low = cand.lower()
        for n in self._short:                      # 実在名 ⊂ 候補(短名のみ)
            if n and n in cand:
                return True, n
        best, best_r = None, 0.0
        clen = len(cand)
        for n in self.names:                       # あいまい一致(長さガード付き)
            ln = len(n)
            if ln < clen * 0.6 or ln > clen * 1.6:
                continue
            r = SequenceMatcher(None, low, n.lower()).ratio()
            if r > best_r:
                best_r, best = r, n
            if best_r >= 0.85:
                break
        if best_r >= self.threshold:
            return True, best
        return False, best                         # 未知(best は参考の近傍名)


def detect_fiction(records: list[dict], name_index: NameIndex,
                   coined: set[str] | None = None) -> list[dict]:
    """未知の固有名詞候補を候補ごとに集約して返す(出現多い順)。

    各要素: {candidate, ctype, count, n_agents, agents, first_step, nearest, examples}。
    count(出現回数)>=2 が上位に来るようソート済み。coined 集合の語は除外する。
    """
    coined = coined or set()
    agg: dict[str, dict] = {}
    for r in records:
        for cand, ctype in _extract_candidates(r["text"]):
            if cand in coined:
                continue
            if ctype == "katakana" and cand in _GENERIC_KATAKANA:
                continue
            if ctype == "ascii" and cand.lower() in _GENERIC_ASCII:
                continue
            known, nearest = name_index.classify(cand)
            if known:
                continue
            a = agg.get(cand)
            if a is None:
                a = agg[cand] = {"candidate": cand, "ctype": ctype, "count": 0,
                                 "agents": set(), "first_step": r["step"],
                                 "nearest": nearest, "examples": []}
            a["count"] += 1
            a["agents"].add(r["agent"])
            a["first_step"] = min(a["first_step"], r["step"])
            if len(a["examples"]) < 3:
                a["examples"].append({
                    "step": r["step"], "agent": r["agent"], "kind": r["kind"],
                    "context": r["text"][:120],
                })
    out = []
    for a in agg.values():
        out.append({
            "candidate": a["candidate"], "ctype": a["ctype"], "count": a["count"],
            "n_agents": len(a["agents"]), "agents": sorted(a["agents"]),
            "first_step": a["first_step"], "nearest": a["nearest"],
            "examples": a["examples"],
        })
    out.sort(key=lambda x: (-x["count"], x["first_step"], x["candidate"]))
    return out


# --------------------------------------------------------------------------- #
# 2. norms(規範発話)
# --------------------------------------------------------------------------- #
_NORM_PATTERNS = [
    (re.compile(r"べき(?:だ|です|だろう|じゃ)?"), "義務 (〜べき)"),
    (re.compile(r"(?:ては|ちゃ)(?:いけ(?:ない|ません)|なら(?:ない|ぬ)|だめ|ダメ)"),
     "禁止 (してはいけない)"),
    (re.compile(r"(?:が|は|って)?(?:普通|当たり前|常識|当然)(?:だ|です|でしょ)?"),
     "社会通念 (普通/当たり前/常識)"),
    (re.compile(r"ものだ(?:ろう)?|ものです"), "規範的一般化 (〜ものだ)"),
    (re.compile(r"マナー"), "マナー"),
    (re.compile(r"ルール(?:を|は)?(?:守|まも)"), "ルール遵守"),
    (re.compile(r"みんな(?:で)?.{0,8}(?:する|している|してる|しよう|すべき)"),
     "同調的一般化 (みんな〜する)"),
    (re.compile(r"(?:した|する)方が(?:いい|良い|よい)"), "推奨 (〜方がいい)"),
    (re.compile(r"(?:should|must|ought to|have to)\b", re.IGNORECASE), "duty (en)"),
    (re.compile(r"(?:everyone|nobody|everybody)\s+\w+", re.IGNORECASE),
     "generalisation (en)"),
]


def detect_norms(records: list[dict]) -> list[dict]:
    """規範発話を1発話1件で検出。最初にマッチしたパターンを採用する。"""
    out: list[dict] = []
    for r in records:
        text = r["text"]
        for pat, label in _NORM_PATTERNS:
            mm = pat.search(text)
            if mm:
                out.append({
                    "step": r["step"], "agent": r["agent"], "kind": r["kind"],
                    "label": label, "matched": mm.group(0), "text": text[:160],
                })
                break
    out.sort(key=lambda x: x["step"])
    return out


# --------------------------------------------------------------------------- #
# 3. attention(語の共伝播)
# --------------------------------------------------------------------------- #
# 形態素解析なしの軽量トークナイザ。**スクリプト境界で切る**のが要点:
#   カタカナ連続 / 2文字以上の漢字連続 / ASCII 語 を content 語として拾い、
#   助詞・語尾のひらがなは落とす。1文字漢字・ひらがなは attention のノイズなので拾わない。
# 限界: 漢字複合語の切れ目は文脈依存(議員選出↔議員/選出)なので長い複合語の一致は取り逃す。
_TOKEN_RE = re.compile(r"[A-Za-z]{2,}|[゠-ヿｦ-ﾟ]{2,}|[一-鿿]{2,}")
_STOPWORDS = {
    # 日本語の高頻度・機能語(創発語としての面白みが薄いもの)
    "今日", "明日", "昨日", "自分", "今回", "今度", "自身", "みんな", "とき", "こと",
    "もの", "ため", "よう", "ところ", "場所", "気持", "時間", "本当", "一緒", "感じ",
    "渋谷", "今朝", "普通", "住民", "世界", "生活", "人々", "地域", "未来", "新しい",
    "the", "and", "for", "with", "you", "are", "this", "that", "from", "have",
    "was", "were", "will", "your", "our", "not", "but",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def whitelist_tokens(names: list[str]) -> set[str]:
    """実在名を同じ流儀でトークン化した除外集合(地名の断片を attention から外す)。"""
    toks: set[str] = set()
    for n in names:
        for t in _tokenize(n):
            toks.add(t)
    return toks


def detect_attention(records: list[dict], exclude_tokens: set[str] | None = None,
                     coined_tokens: set[str] | None = None,
                     window_steps: int = WINDOW_STEPS,
                     min_agents: int = MIN_ATTENTION_AGENTS) -> list[dict]:
    """窓 window_steps 内で min_agents 人以上の別エージェントが使った語を検出。

    ストップワードと exclude_tokens(ホワイトリスト名の断片)は除外。coined_tokens に
    入る語には provenance="coined"、それ以外は "organic" を付す。1語1件。
    """
    exclude = (exclude_tokens or set()) | _STOPWORDS
    coined_tokens = coined_tokens or set()
    occ: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(step, agent)]
    for r in records:
        for tok in set(_tokenize(r["text"])):        # 1発話内の同語は1回だけ
            if tok in exclude:
                continue
            occ[tok].append((r["step"], r["agent"]))

    out: list[dict] = []
    for term, occs in occ.items():
        occs.sort()
        for i, (s_i, _) in enumerate(occs):
            window = [(s, a) for (s, a) in occs if s_i <= s <= s_i + window_steps]
            agents = {a for (_, a) in window}
            if len(agents) >= min_agents:
                out.append({
                    "term": term,
                    "first_step": s_i,
                    "window_end_step": s_i + window_steps,
                    "n_agents": len(agents),
                    "agents": sorted(agents),
                    "occurrences": len(window),
                    "provenance": "coined" if term in coined_tokens else "organic",
                })
                break                                # 1語1件
    out.sort(key=lambda x: (-x["n_agents"], -x["occurrences"], x["first_step"]))
    return out


# --------------------------------------------------------------------------- #
# 4. grounding(作話の接地率メトリクス。reality-levers P2 / L2 観測のみ)
# --------------------------------------------------------------------------- #
# 発話(speak/sns_post/dm)中の固有名詞候補を、実在名(地図・名簿・組織台帳)/シミュ内実在
# (創発名=造語・店名・コミュニティ・制度)/ 作話(どちらでもない)の3値に分類し、日次で集計する。
# 「接地率 rate」= 実在名 / 固有名詞総数。**創発名を作話に混ぜない**のが本メトリクスの肝。
# 固有名詞抽出は detect_fiction と同じ保守的ヒューリスティクス(形態素解析なし)を流用する。
_GROUNDING_KINDS = ("speak", "sns_post", "dm")           # 内省(reflect)は対象外


def _classify_proper(cand: str, ctype: str, real_index: "NameIndex",
                     sim_index: "NameIndex | None") -> str:
    """固有名詞候補を grounded(実在名)/ sim_coined(シミュ内実在)/ fiction(作話)に分類。

    優先順は 実在名 > シミュ内実在 > 作話。実在名照合を先に当てることで、地図/台帳にある
    名前を誤って「創発名」に落とさない。"""
    known, _ = real_index.classify(cand)
    if known:
        return "grounded"
    if sim_index is not None:
        s_known, _ = sim_index.classify(cand)
        if s_known:
            return "sim_coined"
    return "fiction"


def compute_grounding(records: list[dict], real_index: "NameIndex",
                      sim_index: "NameIndex | None" = None,
                      steps_per_day: int = STEPS_PER_DAY) -> dict:
    """発話中の固有名詞の接地率を日次で集計する。

    records: extract_text_records の {step, agent, kind, text}(reflect 混在可。内部で
             speak/sns_post/dm のみに絞る)。real_index=実在名、sim_index=シミュ内実在の照合器。

    返り値:
      daily     : 日ごとの [{day, n_utter, n_proper, n_grounded, n_sim_coined,
                  n_fiction, rate, rate_incl_sim, fabrication_rate}]。n_proper=0 の日は
                  rate 系を None(=データ不足)にする。
      totals    : 全期間の同集計(rate 系は分母0で None)。
      fiction   : 作話候補の集約 [{candidate, ctype, count, n_agents, first_step, examples}]
                  (出現多い順。レポート・作話例の引用に使う)。
    決定論: records の順序に依存しない集約(day/候補で確定ソート)。
    """
    spoken = [r for r in records if r["kind"] in _GROUNDING_KINDS]

    def _blank() -> dict:
        return {"n_utter": 0, "n_proper": 0, "n_grounded": 0,
                "n_sim_coined": 0, "n_fiction": 0}

    per_day: dict[int, dict] = defaultdict(_blank)
    fic: dict[str, dict] = {}                            # 作話候補の集約

    for r in spoken:
        d = r["step"] // steps_per_day
        per_day[d]["n_utter"] += 1
        for cand, ctype in _extract_candidates(r["text"]):
            # 一般語(固有名詞ではない普通名詞)は分母から除外する(detect_fiction と同基準)。
            if ctype == "katakana" and cand in _GENERIC_KATAKANA:
                continue
            if ctype == "ascii" and cand.lower() in _GENERIC_ASCII:
                continue
            cls = _classify_proper(cand, ctype, real_index, sim_index)
            per_day[d]["n_proper"] += 1
            per_day[d][f"n_{cls}"] += 1
            if cls == "fiction":
                a = fic.get(cand)
                if a is None:
                    a = fic[cand] = {"candidate": cand, "ctype": ctype, "count": 0,
                                     "agents": set(), "first_step": r["step"],
                                     "examples": []}
                a["count"] += 1
                a["agents"].add(r["agent"])
                a["first_step"] = min(a["first_step"], r["step"])
                if len(a["examples"]) < 3:
                    a["examples"].append({
                        "step": r["step"], "agent": r["agent"], "kind": r["kind"],
                        "context": r["text"][:120],
                    })

    def _rates(c: dict) -> dict:
        np_ = c["n_proper"]
        row = dict(c)
        if np_ > 0:
            row["rate"] = round(c["n_grounded"] / np_, 4)
            row["rate_incl_sim"] = round(
                (c["n_grounded"] + c["n_sim_coined"]) / np_, 4)
            row["fabrication_rate"] = round(c["n_fiction"] / np_, 4)
        else:                                            # 0件日 = データ不足
            row["rate"] = row["rate_incl_sim"] = row["fabrication_rate"] = None
        return row

    daily = []
    for d in sorted(per_day):
        row = _rates(per_day[d])
        row = {"day": d, **row}
        daily.append(row)

    totals_c = _blank()
    for c in per_day.values():
        for k in totals_c:
            totals_c[k] += c[k]
    totals = _rates(totals_c)

    fiction = []
    for a in fic.values():
        fiction.append({
            "candidate": a["candidate"], "ctype": a["ctype"], "count": a["count"],
            "n_agents": len(a["agents"]), "first_step": a["first_step"],
            "examples": a["examples"],
        })
    fiction.sort(key=lambda x: (-x["count"], x["first_step"], x["candidate"]))

    return {"daily": daily, "totals": totals, "fiction": fiction,
            "n_days": len(daily), "n_utterances": len(spoken)}


_GROUNDING_COLS = ["run", "day", "n_utter", "n_proper", "n_grounded",
                   "n_sim_coined", "n_fiction", "rate", "rate_incl_sim",
                   "fabrication_rate"]


def _grounding_schema():
    import pyarrow as pa
    f = pa.field
    return pa.schema([
        f("run", pa.string()), f("day", pa.int64()), f("n_utter", pa.int64()),
        f("n_proper", pa.int64()), f("n_grounded", pa.int64()),
        f("n_sim_coined", pa.int64()), f("n_fiction", pa.int64()),
        f("rate", pa.float64()), f("rate_incl_sim", pa.float64()),
        f("fabrication_rate", pa.float64()),
    ])


def write_grounding_parquet(run_name: str, grounding: dict, out_dir: str) -> str:
    """接地率の日次系列を panel/grounding_rate.parquet に書き出す(pyarrow 直書き)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    cols: dict[str, list] = {c: [] for c in _GROUNDING_COLS}
    for row in grounding["daily"]:
        cols["run"].append(run_name)
        cols["day"].append(int(row["day"]))
        cols["n_utter"].append(int(row["n_utter"]))
        cols["n_proper"].append(int(row["n_proper"]))
        cols["n_grounded"].append(int(row["n_grounded"]))
        cols["n_sim_coined"].append(int(row["n_sim_coined"]))
        cols["n_fiction"].append(int(row["n_fiction"]))
        cols["rate"].append(row["rate"])
        cols["rate_incl_sim"].append(row["rate_incl_sim"])
        cols["fabrication_rate"].append(row["fabrication_rate"])
    path = os.path.join(out_dir, "grounding_rate.parquet")
    pq.write_table(pa.Table.from_pydict(cols, schema=_grounding_schema()), path)
    return path


# --------------------------------------------------------------------------- #
# 実行・出力
# --------------------------------------------------------------------------- #
def analyze(run_dir: str) -> dict:
    events = load_events(run_dir)
    agents = load_agents(run_dir)
    map_data = load_map_for_run(run_dir)
    orgs = load_orgs_for_run(run_dir)
    names = whitelist_names(map_data, agents, orgs)
    coined = collect_coined(events)
    coined_texts = {c["text"] for c in coined}
    coined_toks = whitelist_tokens([c["text"] for c in coined])

    # シミュ内実在(創発名)の照合台帳: 造語 + 店名 + コミュニティ + 制度名。
    sim_names = collect_sim_names(events)
    sim_index = NameIndex(sorted({s["text"] for s in sim_names}))

    records = extract_text_records(events)
    name_index = NameIndex(names)
    fiction = detect_fiction(records, name_index, coined_texts)
    norms = detect_norms(records)
    attention = detect_attention(records, whitelist_tokens(names), coined_toks)
    grounding = compute_grounding(records, name_index, sim_index)

    return {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "n_text_records": len(records),
        "n_whitelist_names": len(names),
        "n_sim_names": len({s["text"] for s in sim_names}),
        "map_loaded": map_data is not None,
        "orgs_loaded": orgs is not None,
        "model_backend": load_model_backend(run_dir),
        "fiction": fiction,
        "norms": norms,
        "attention": attention,
        "coined_vocab": coined,
        "grounding": grounding,
    }


def _pick_run(arg_run: str | None) -> str:
    """--run 指定 or l1_events.parquet を持つ最新ランを返す。"""
    if arg_run:
        d = arg_run if os.path.isabs(arg_run) else os.path.join(RUNS_ROOT, arg_run)
        if not os.path.isfile(os.path.join(d, "l1_events.parquet")):
            raise SystemExit(f"[detect] l1_events.parquet が無い: {d}")
        return d
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit(f"[detect] runs が無い: {RUNS_ROOT}")
    cands = []
    for name in os.listdir(RUNS_ROOT):
        pq_path = os.path.join(RUNS_ROOT, name, "l1_events.parquet")
        if os.path.isfile(pq_path):
            cands.append((os.path.getmtime(pq_path), os.path.join(RUNS_ROOT, name)))
    if not cands:
        raise SystemExit("[detect] l1_events.parquet を持つランが無い")
    cands.sort(reverse=True)
    return cands[0][1]


def _norm_label_counts(norms: list[dict]) -> list[tuple[str, int]]:
    return Counter(n["label"] for n in norms).most_common()


def build_report(res: dict) -> str:
    L: list[str] = []
    L.append(f"# 創発の後付けテキスト検出レポート — {res['run']}\n")
    L.append("> 保守的なヒューリスティクス(形態素解析なし)。各命中は断定ではなく"
             "**観測候補**です。fiction は一般語・誤字も混じり、norms/attention も"
             "取りこぼし(再現率)があります。\n")
    L.append(f"- テキストイベント数: **{res['n_text_records']}**"
             f"(speak/sns_post/dm/reflect[written_back])")
    L.append(f"- 実在名ホワイトリスト: **{res['n_whitelist_names']}** 件"
             f"(地図={'読込OK' if res['map_loaded'] else '未読込→精度低下'})")
    L.append(f"- 検出: fiction **{len(res['fiction'])}** 候補 / "
             f"norms **{len(res['norms'])}** 件 / "
             f"attention **{len(res['attention'])}** 語 / "
             f"造語 **{len(res['coined_vocab'])}** 件\n")

    # --- fiction ---
    L.append("## 1. 架空固有名詞 (fiction)")
    L.append("カタカナ連続・「」引用・ASCII 固有名詞のうち、実在名に一致しなかったもの。"
             "出現2回以上を優先表示。\n")
    multi = [f for f in res["fiction"] if f["count"] >= 2]
    show = multi[:20] if multi else res["fiction"][:20]
    if not show:
        L.append("_(候補なし)_\n")
    else:
        L.append("| 候補 | 種別 | 出現 | 人数 | 初出step | 文脈例 |")
        L.append("|---|---|---:|---:|---:|---|")
        for f in show:
            ctx = f["examples"][0]["context"] if f["examples"] else ""
            ctx = ctx.replace("|", "/").replace("\n", " ")
            L.append(f"| {f['candidate']} | {f['ctype']} | {f['count']} | "
                     f"{f['n_agents']} | {f['first_step']} | {ctx} |")
        if multi and len(res["fiction"]) > len(multi):
            L.append(f"\n_(ほか出現1回の候補 {len(res['fiction']) - len(multi)} 件は "
                     "emergence.json に収録)_")
    L.append("")

    # --- norms ---
    L.append("## 2. 規範発話 (norms)")
    counts = _norm_label_counts(res["norms"])
    if not counts:
        L.append("_(検出なし)_\n")
    else:
        L.append("パターン別件数: " + ", ".join(f"{lb}×{c}" for lb, c in counts) + "\n")
        L.append("| step | agent | 種別 | 該当 | 発話 |")
        L.append("|---:|---:|---|---|---|")
        for n in res["norms"][:15]:
            tx = n["text"].replace("|", "/").replace("\n", " ")
            L.append(f"| {n['step']} | {n['agent']} | {n['label']} | "
                     f"{n['matched']} | {tx} |")
    L.append("")

    # --- attention ---
    L.append("## 3. 語の共伝播 (collective attention)")
    L.append(f"窓 {WINDOW_STEPS}step(4時間)以内に {MIN_ATTENTION_AGENTS} 人以上の別"
             "エージェントが使った語(ストップワード・地名断片は除外)。\n")
    att = res["attention"]
    if not att:
        L.append("_(共伝播語なし)_\n")
    else:
        L.append("| 語 | 人数 | 出現 | 窓(step) | provenance | agents |")
        L.append("|---|---:|---:|---|---|---|")
        for a in att[:20]:
            L.append(f"| {a['term']} | {a['n_agents']} | {a['occurrences']} | "
                     f"{a['first_step']}–{a['window_end_step']} | {a['provenance']} | "
                     f"{a['agents']} |")
    L.append("")

    # --- coined ---
    L.append("## 4. 正規造語 (coined)")
    if not res["coined_vocab"]:
        L.append("_(label_coin / vocab_coin イベントなし)_")
    else:
        L.append("| step | agent | item_id | 語 |")
        L.append("|---:|---:|---|---|")
        for c in res["coined_vocab"][:30]:
            L.append(f"| {c['step']} | {c['agent']} | {c['item_id']} | {c['text']} |")
    L.append("")

    # --- grounding(作話の接地率) ---
    L.extend(_grounding_section(res))
    return "\n".join(L)


def _grounding_section(res: dict) -> list[str]:
    """作話の接地率(grounding)節を組み立てる。build_report から呼ぶ。"""
    g = res.get("grounding") or {}
    tot = g.get("totals", {})
    L: list[str] = []
    L.append("## 5. 作話の接地率 (grounding)")
    L.append("発話(speak/sns_post/dm)中の固有名詞候補を **実在名**(地図POI/建物/駅路線・"
             "ペルソナ名簿・組織台帳)/ **シミュ内実在**(創発名=造語・店名・コミュニティ・制度)/ "
             "**作話**(どちらでもない)の3値に分類し日次集計。`rate`=実在名/固有名詞総数。"
             "**創発名は作話に数えない**(シミュ内で実体を持つため)。")
    # 誠実性の注記
    L.append("")
    L.append("> **誠実性の注記**: 固有名詞抽出は形態素解析なしの保守的ヒューリスティクス"
             "(カタカナ連続・「」引用・ASCII 固有名詞)です。台帳に無いカタカナ一般名詞が"
             "作話へ混入しうる(過大評価)一方、漢字の固有名や助詞連結は取りこぼす(過小評価)。"
             "`n_proper=0` の日は **データ不足** として rate を空欄(None)にします。")
    be = res.get("model_backend")
    if be and str(be).lower() in ("mock", "memory", "template"):
        L.append(f"> **注意**: このランの model.backend=`{be}`(mock 系)です。発話がテンプレ"
                 "由来のため実在名を多く含み、接地率が**無意味に高く**出ます。実 LLM ランと"
                 "同列には扱えません。")
    L.append(f"> 台帳: 実在名 **{res.get('n_whitelist_names', 0)}** 件"
             f"(組織台帳={'あり' if res.get('orgs_loaded') else 'なし'})/ "
             f"シミュ内実在(創発名)**{res.get('n_sim_names', 0)}** 件。")
    L.append("")

    # 全期間サマリ
    def _pct(x):
        return "—" if x is None else f"{x * 100:.1f}%"
    if tot.get("n_proper"):
        L.append(f"- 全期間: 固有名詞 **{tot['n_proper']}** / 実在名 {tot['n_grounded']} / "
                 f"シミュ内実在 {tot['n_sim_coined']} / 作話 {tot['n_fiction']}")
        L.append(f"- 接地率(実在名) **{_pct(tot['rate'])}** / "
                 f"実在名+シミュ内実在 {_pct(tot['rate_incl_sim'])} / "
                 f"作話率 {_pct(tot['fabrication_rate'])}\n")
    else:
        L.append("_(固有名詞候補が無く接地率を算定できません=データ不足)_\n")

    # 日次系列
    daily = g.get("daily", [])
    if daily:
        L.append("| day | 発話 | 固有名詞 | 実在名 | シミュ内実在 | 作話 | "
                 "接地率 | 作話率 |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in daily:
            L.append(f"| {row['day']} | {row['n_utter']} | {row['n_proper']} | "
                     f"{row['n_grounded']} | {row['n_sim_coined']} | {row['n_fiction']} | "
                     f"{_pct(row['rate'])} | {_pct(row['fabrication_rate'])} |")
        L.append("")

    # 作話の実例
    fic = g.get("fiction", [])
    if fic:
        L.append("### 作話の実例(出現多い順)")
        L.append("| 候補 | 種別 | 出現 | 人数 | 初出step | 文脈例 |")
        L.append("|---|---|---:|---:|---:|---|")
        for f in fic[:15]:
            ctx = f["examples"][0]["context"] if f["examples"] else ""
            ctx = ctx.replace("|", "/").replace("\n", " ")
            L.append(f"| {f['candidate']} | {f['ctype']} | {f['count']} | "
                     f"{f['n_agents']} | {f['first_step']} | {ctx} |")
        L.append("")
    L.append(f"> パネル: `panel/grounding_rate.parquet`(日次系列)。")
    L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(
        description="創発の後付けテキスト検出(fiction/norms/attention・読み出し専用)")
    ap.add_argument("--run", default=None,
                    help="ラン名(既定: l1_events.parquet を持つ最新ラン)")
    args = ap.parse_args()

    run_dir = _pick_run(args.run)
    res = analyze(run_dir)
    g = res["grounding"]["totals"]
    print(f"[detect] run={res['run']}  text_records={res['n_text_records']}  "
          f"whitelist={res['n_whitelist_names']}(map={'ok' if res['map_loaded'] else 'NG'})")
    print(f"[detect] fiction={len(res['fiction'])}  norms={len(res['norms'])}  "
          f"attention={len(res['attention'])}  coined={len(res['coined_vocab'])}")
    _grate = "—" if g["rate"] is None else f"{g['rate'] * 100:.1f}%"
    print(f"[detect] grounding: proper={g['n_proper']} grounded={g['n_grounded']} "
          f"sim_coined={g['n_sim_coined']} fiction={g['n_fiction']} rate={_grate}")

    json_path = os.path.join(run_dir, "emergence.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    md_path = os.path.join(run_dir, "emergence_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(build_report(res))
    # 接地率の日次系列を panel/ に書き出す(pyarrow 直書き)。
    panel_dir = os.path.join(run_dir, "panel")
    gp_path = write_grounding_parquet(res["run"], res["grounding"], panel_dir)
    print(f"[detect] -> {json_path}")
    print(f"[detect] -> {md_path}")
    print(f"[detect] -> {gp_path}")

    # コンソールにも上位の見どころを少し
    if res["fiction"]:
        print("\n[fiction top]")
        for f in res["fiction"][:8]:
            print(f"  '{f['candidate']}' x{f['count']} ({f['ctype']}) "
                  f"step{f['first_step']}: {f['examples'][0]['context'][:60]}")
    if res["attention"]:
        print("\n[attention top]")
        for a in res["attention"][:8]:
            print(f"  '{a['term']}' agents={a['n_agents']} occ={a['occurrences']} "
                  f"[{a['provenance']}]")


if __name__ == "__main__":
    main()
