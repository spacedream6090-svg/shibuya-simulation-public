"""ペルソナ来歴(backstory)の**事前**生成(サイドカー方式・ラン中は読むだけ)。

data/persona_pool_v2(100万人)の各ペルソナへ、LLM が書いた過去情報 2〜3文
(日本語・目安 80〜140字)を**事前に**作って別ディレクトリへ保存する。
プール本体は 1 バイトも触らない(読むだけ)。ラン中はこの凍結成果物を読むだけなので、
**ランの決定論は不変**(生成時の LLM 呼び出しはランの乱数 stream に一切関与しない)。

出力
----
    <out>/<layer>.jsonl.gz   1 行 = {"pid","backstory","model","seed","prompt_sha","elements"
                                     [,"retry" | ,"rescue"]}
    <out>/<layer>.failed.jsonl  検品に落ちた pid と理由(再実行で再挑戦される。追記)
    <out>/meta.json          生成 config の全記録(温度・プロンプト版・モデル・層別件数・時刻・重複率)

モデル群(2 グループ・汎用)
--------------------------
サーバ群を **core / mass の 2 グループ**で受ける。どのモデルを載せるかはスクリプトに
固定しない(起動側の served name をそのまま `model` として送る)。

    --core-servers  ...   --core-model <served_name>   # コア層(既定 L1,L3)
    --mass-servers  ...   --mass-model <served_name>   # 残りの層
    --core-layers L1,L3                                # 層割当はフラグで可変
    --layers L2,L4,L5                                  # ★包含フィルタ(既定=全層)

割当は pid の安定ハッシュで sticky(同じ pid は常に同じサーバ = prefix cache が効く)。

**片側運転(2 段運用)**: `--layers` で走る層が片方のグループだけなら、もう一方の
servers/model は**未指定で良い**(`/v1/models` プローブもその段で使う群だけ引く)。
層がルーティングされるのに設定が無いグループがあれば、黙って走らせずに起動時エラーで止める。

決定論
------
  * 要素選択(下記 ELEMENTS から 2 つ)は pid の blake2b ハッシュ = 個体ごとに組合せが変わる
    (テンプレ化・mode collapse 対策)。
  * リクエスト seed = blake2b(seed, pid, "backstory"[, "retry"]) & 0x7fffffff
    (src/society/llm/vllm.py の β11 `stable_request_seed` と同型。近似再現性のため)。
  * temperature は既定 0.9・top_p は**送らない**(サーバ既定に委ねる)。

中断安全(resumable)
--------------------
出力済み pid はスキップする。異常終了で gz の末尾が壊れていた場合は、読めた行までを
書き直してから追記を再開する(壊れた尾を引きずらない)。検品に落ちた pid は本体へ書かない
= 次の実行で自動的に引き直される。連続失敗が `--abort-after-failures` に達したら中止する
(無人運転でサーバが落ちたまま何十万件も空振りするのを防ぐ)。

実行コマンド例(**サーバー側で親が叩く**。ローカルでは実行しない)
-----------------------------------------------------------------
    python scripts/build_persona_backstory.py \
        --pool data/persona_pool_v2 --out data/persona_backstory_v2 \
        --core-servers http://localhost:8005,http://localhost:8006 --core-model qwen3:32b \
        --mass-servers http://localhost:8000,http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004 \
        --mass-model qwen3:8b \
        --core-layers L1,L3 --concurrency 96

    # 疎通確認(各層 20 件だけ)
    python scripts/build_persona_backstory.py --limit 20 --out data/persona_backstory_smoke ...

2 段運用(同じ --out へ書き足す。層ごとに別ファイルなので衝突しない)
--------------------------------------------------------------------
  ① mass 段(いま動いているサーバだけで L2+L4+L5): --layers L2,L4,L5 + --mass-* のみ
  ② core 段(大きいモデルが起動してから L1+L3): --layers L1,L3 + --core-* のみ

救済(`--rescue N`)= 恒常失敗の回収
-----------------------------------
`<layer>.failed.jsonl` に残った pid **だけ**を対象に、試行 k=1..N で

  * seed 塩 …… `blake2b(seed, pid, "backstory", "rescue<k>")`(本番の "" / "retry" とも別列)
  * 要素組合せ …… `pick_elements(pid, seed, attempt=k)` = **別の切り口**で書かせる
  * 字数の駄目押し 1 行(実測の恒常失敗理由 = short 対策)

を順に試す。成功したら本体 jsonl.gz へ追記して failed から外し、**全 attempt 落ちた個体
だけ**が failed に残る(理由と `rescue_attempts` を更新)。全部回収できたら failed
ファイル自体を消す。seed 列も要素列も (pid, k) の純関数なので、再実行すれば同じ列を辿る。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import threading
import time
import urllib.request
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

SCHEMA_VERSION = "persona-backstory-1.0"
#: プロンプト規約の版。文面を変えたら必ず上げる(prompt_sha と併せて来歴になる)。
PROMPT_VERSION = "bs-1"

#: 受理する本文長の帯(字)。この外は 1 度だけ再生成する。
LEN_MIN = 40
LEN_MAX = 200
#: プロンプトで指示する目安(帯より狭い = 帯は「落とさない」ための余白)。
TARGET_MIN = 80
TARGET_MAX = 140

#: 来歴に含める要素。pid ハッシュで 2 つ選ぶ(組合せ 5×4=20 通り)。
ELEMENTS: tuple[tuple[str, str], ...] = (
    ("幼少期の土地", "どこで育ったか(地方や街の雰囲気の程度で)"),
    ("前職や転機", "いまの立場に至るまでの仕事や進路の変化。年少者なら引っ越し・進学・習い事の転機に読み替える"),
    ("趣味嗜好", "休みの日の過ごし方や好きなもの"),
    ("人間関係の癖", "人との距離の取り方・付き合い方の傾向"),
    ("いま気にしていること", "最近の暮らしで気にしている小さな関心事"),
)

SYSTEM_PROMPT = (
    "あなたは架空の人物の来歴を書く日本語の書き手です。次の規則を必ず守ってください。\n"
    "- 対象は完全に架空の人物です。実在の人名・企業名・店舗名・学校名・住所・電話番号・"
    "メールアドレス・SNSアカウントは書かない。\n"
    "- 与えられた骨格(氏名・年齢・性別・職業・立場・住まい)と矛盾する過去を書かない。\n"
    "- 未就学児や小中学生には職歴を与えない。家庭・園や学校・遊びの話にする。\n"
    "- 日本語の地の文だけ。箇条書き・見出し・JSON・英語・絵文字・台詞は使わない。\n"
    "- 「あなた」「私」は使わない。氏名を主語にするか主語を省く。\n"
    f"- 全体で2〜3文・{TARGET_MIN}〜{TARGET_MAX}字。前置きや要約は書かず、本文だけを出力する。"
)

#: 応答から剥がすもの(qwen3 の思考ブロック・先頭ラベル・囲みの括弧)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_LEAD_LABEL = re.compile(r"^\s*(?:【[^】]{1,12}】|(?:来歴|経歴|背景|プロフィール|回答|出力)\s*[:：])\s*")
_WS = re.compile(r"\s+")
_ASCII_RUN = re.compile(r"[A-Za-z]{10,}")
# ★リテラルを書かない(この文字種はリポジトリで禁則)= コードポイント指定で持つ
_CYRILLIC = re.compile("[%s-%s]" % (chr(0x0400), chr(0x052F)))
_JP_CHARS = re.compile(r"[぀-ヿ㐀-䶿一-鿿　-〿0-9]")
#: 重複検出の正規化(記号・空白を落とした素の文字列)
_NON_WORD = re.compile(r"[\s、。，．・「」『』()()!?！？…—\-]")

_SEP = b"\x1f"
_SEED_MASK = 0x7FFFFFFF
_LAYERS = ("L1", "L2", "L3", "L4", "L5")


# ---------------------------------------------------------------- 安定ハッシュ
def _stable_h(*fields: str) -> int:
    """材料の blake2b(8 バイト)→ 非負整数。プロセス跨ぎ・resume で安定(`hash()` は使わない)。"""
    h = hashlib.blake2b(digest_size=8)
    for f in fields:
        h.update(str(f).encode("utf-8"))
        h.update(_SEP)
    return int.from_bytes(h.digest(), "big")


def stable_seed(seed: int, pid: str, purpose: str = "backstory",
                ordinal: str = "") -> int:
    """リクエストへ送る seed。β11(vllm.stable_request_seed)と同型の 31bit 非負整数。"""
    return _stable_h(str(int(seed)), pid, purpose, ordinal) & _SEED_MASK


def pick_elements(pid: str, seed: int, attempt: int = 0) -> tuple[int, int]:
    """pid ハッシュで ELEMENTS から**相異なる 2 つ**を選ぶ(順序も個体で変わる)。

    `attempt`(救済の試行番号 1..N)を混ぜると**別の切り口**の組合せに回る。
    ★attempt=0(本番パス)は材料を 1 つも足さない = 従来と同一値(決定論の保存)。
    """
    material = ("elements", pid) if not attempt else ("elements", pid, f"a{int(attempt)}")
    h = _stable_h(str(int(seed)), *material)
    n = len(ELEMENTS)
    i = h % n
    j = (i + 1 + (h // n) % (n - 1)) % n      # i != j を構成的に保証
    return i, j


def pick_server(pid: str, servers: list[str]) -> str:
    """pid の安定ハッシュでサーバへ sticky 割当(同じ pid は常に同じサーバ = prefix cache)。"""
    return servers[_stable_h("server", pid) % len(servers)]


# ---------------------------------------------------------------- プロンプト
_PRESENCE_LINE = {
    "resident": "渋谷区に住んでいる住民",
    "workday_shift": "区外から渋谷へ通勤している",
    "cadence": "区外から渋谷へ定期的に通っている",
    "stochastic": "渋谷を訪れる来街者",
    "duty": "渋谷で公共の持ち場に就いている",
}


def _persona_core(rec: dict) -> str:
    """プール本体の persona 文から、末尾の発話指示だけを落とした人物メモ。"""
    text = str(rec.get("persona", "") or "")
    return text.replace("自分の言葉で自然に、短く話す。", "").strip()


def skeleton_lines(rec: dict) -> list[str]:
    """LLM へ渡す骨格(**制約**)。存在する欄だけを書く=捏造しない。"""
    out: list[str] = []
    occ = str(rec.get("occupation", "") or "")
    role = str(rec.get("role", "") or "")
    if occ:
        out.append(f"職業: {occ}" + (f"({role})" if role and role != occ else ""))
    elif role:
        out.append(f"役割: {role}")
    ind = str(rec.get("industry_major", "") or "")
    if ind:
        out.append(f"業種: {ind}")
    stand = _PRESENCE_LINE.get(str(rec.get("presence", "")), "")
    line = str(rec.get("residence_line", "") or "")
    if stand:
        out.append("立場: " + stand + (f"(住まい: {line})" if line else ""))
    if rec.get("is_foreign"):
        out.append("属性: 訪日外国人")
    purpose = str(rec.get("visit_purpose", "") or "")
    if purpose:
        out.append(f"来街目的: {purpose}")
    stage = rec.get("school_stage")
    if stage:
        out.append(f"通学先: {stage}")
    post = str(rec.get("post", "") or "")
    if post:
        out.append(f"持ち場: {post}")
    htype = str(rec.get("household_type", "") or "")
    hrole = str(rec.get("household_role", "") or "")
    if htype or hrole:
        out.append("世帯: " + "・".join(x for x in (htype, hrole) if x))
    emp = str(rec.get("employment", "") or "")
    rank = str(rec.get("rank", "") or "")
    if emp or rank:
        out.append("就業: " + "・".join(x for x in (emp, rank) if x))
    return out


def build_prompt(rec: dict, seed: int,
                 attempt: int = 0) -> tuple[str, str, tuple[str, str]]:
    """(system, user, 選ばれた要素ラベル 2 つ)。同じ record・同じ seed なら常に同一。

    `attempt`(救済の試行番号 1..N)が入ると、要素の組合せが回り、字数の駄目押しが 1 行
    足される(= 別の切り口・別の長さ圧で書かせる)。★attempt=0 は従来と 1 バイト同一。
    """
    pid = str(rec["id"])
    i, j = pick_elements(pid, seed, attempt)
    (la, ha), (lb, hb) = ELEMENTS[i], ELEMENTS[j]
    gender = {"男": "男性", "女": "女性"}.get(str(rec.get("gender", "")), str(rec.get("gender", "")))
    head = f"【人物】{rec.get('name', '')}({rec.get('age', '')}歳・{gender})"
    body = "\n".join(f"- {ln}" for ln in skeleton_lines(rec))
    memo = _persona_core(rec)
    user = (
        f"{head}\n"
        f"【骨格(この条件と矛盾させない)】\n{body}\n"
        + (f"【人物メモ】{memo}\n" if memo else "")
        + "【必ず含める要素】\n"
        f"1. {la}({ha})\n"
        f"2. {lb}({hb})\n"
        f"【出力】この人物の来歴を日本語で2〜3文・{TARGET_MIN}〜{TARGET_MAX}字。本文だけ。"
        # 救済パスだけの駄目押し(恒常失敗の実測理由 = short = 1 文で切り上げてしまう)
        + (f"\n【字数厳守】2 文以上書き、全体で必ず {TARGET_MIN} 字以上 {TARGET_MAX} 字以内。"
           if attempt else "")
    )
    return SYSTEM_PROMPT, user, (la, lb)


def prompt_sha(system: str, user: str) -> str:
    """プロンプト(版込み)の指紋。同 pid → 同値・文面を変えれば必ず変わる。"""
    h = hashlib.sha256()
    for part in (PROMPT_VERSION, system, user):
        h.update(part.encode("utf-8"))
        h.update(_SEP)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- 検品
def normalize(text: str) -> str:
    """応答 → 保存する本文(思考ブロック・ラベル・改行・尻切れ文を落とす)。"""
    t = _THINK_BLOCK.sub("", str(text or ""))
    t = t.replace("<think>", "").replace("</think>", "")
    t = _LEAD_LABEL.sub("", t.strip())
    t = _WS.sub("", t)                       # 日本語なので空白は詰める
    t = t.strip("「」『』\"'“”")
    if "。" in t:                             # max_tokens で切れた尻切れ文を捨てる
        t = t[:t.rindex("。") + 1]
    if len(t) > LEN_MAX and "。" in t[:LEN_MAX + 1]:
        t = t[:t[:LEN_MAX + 1].rindex("。") + 1]
    return t


def check(text: str) -> str | None:
    """検品。問題があれば理由文字列、無ければ None。"""
    if not text:
        return "empty"
    if "{" in text or "}" in text or '":' in text:
        return "json"
    if _CYRILLIC.search(text):
        return "charset"
    jp = len(_JP_CHARS.findall(text))
    if jp / max(1, len(text)) < 0.5 or _ASCII_RUN.search(text):
        return "english"
    if len(text) < LEN_MIN:
        return "short"
    if len(text) > LEN_MAX:
        return "long"
    return None


def dup_key(text: str) -> int:
    """重複検出用の正規化ハッシュ(記号・空白を落とした素の文字列)。"""
    return _stable_h("dup", _NON_WORD.sub("", text))


# ---------------------------------------------------------------- HTTP
class Group:
    """1 モデル群(サーバ列 + served name)。core / mass の 2 つを使う。"""

    def __init__(self, name: str, servers: list[str], model: str):
        self.name = name
        self.servers = [s.rstrip("/") for s in servers if s.strip()]
        self.model = model

    def url_for(self, pid: str) -> str:
        return pick_server(pid, self.servers)


def _post(url: str, body: dict, timeout_s: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:   # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def probe_models(servers: list[str], timeout_s: float = 10.0) -> list[dict]:
    """各サーバの /v1/models を 1 度だけ引いて来歴に残す(失敗は正直に error で残す)。"""
    out: list[dict] = []
    for url in servers:
        entry: dict = {"port": url.rsplit(":", 1)[-1]}
        try:
            req = urllib.request.Request(f"{url}/v1/models")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            first = (data.get("data") or [{}])[0]
            entry.update({k: first.get(k) for k in ("id", "root", "max_model_len")
                          if first.get(k) is not None})
        except Exception as exc:                                   # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}"
        out.append(entry)
    return out


class _AbortRun(RuntimeError):
    """連続失敗が閾値を超えた(= サーバが落ちている疑い)。無人運転を早めに止める。"""


class Runner:
    """1 件生成(検品つき・失敗は 1 度だけ再生成)。スレッドから呼ばれる。"""

    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg

    def _call(self, group: Group, pid: str, system: str, user: str,
              seed: int) -> tuple[str, int]:
        body = {
            "model": group.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_tokens),
            "chat_template_kwargs": {"enable_thinking": False},
            "seed": int(seed),
        }
        urls = [group.url_for(pid)]
        if len(group.servers) > 1:                # 1 度だけ隣のサーバへ逃がす
            urls.append(group.servers[(group.servers.index(urls[0]) + 1) % len(group.servers)])
        last = ""
        for attempt in range(int(self.cfg.http_retries) + 1):
            url = urls[min(attempt, len(urls) - 1)]
            try:
                data = _post(url, body, float(self.cfg.timeout_s))
                choice = (data.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content", "") or ""
                tokens = int((data.get("usage") or {}).get("completion_tokens", 0) or 0)
                return text, tokens
            except Exception as exc:                               # noqa: BLE001
                last = f"http:{type(exc).__name__}"
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
        return f"__error__{last}", 0

    def run(self, rec: dict, group: Group) -> dict:
        pid = str(rec["id"])
        system, user, elements = build_prompt(rec, int(self.cfg.seed))
        sha = prompt_sha(system, user)
        tokens = 0
        for ordinal in ("", "retry"):
            seed = stable_seed(int(self.cfg.seed), pid, "backstory", ordinal)
            raw, tok = self._call(group, pid, system, user, seed)
            tokens += tok
            if raw.startswith("__error__"):
                reason = raw[len("__error__"):]
                continue
            text = normalize(raw)
            reason = check(text)
            if reason is None:
                out = {"pid": pid, "backstory": text, "model": group.model,
                       "seed": seed, "prompt_sha": sha, "elements": list(elements)}
                if ordinal:
                    out["retry"] = 1
                return {"ok": True, "line": out, "tokens": tokens,
                        "retried": bool(ordinal)}
        return {"ok": False, "pid": pid, "reason": reason or "unknown",
                "prompt_sha": sha, "tokens": tokens, "retried": True}

    def run_rescue(self, rec: dict, group: Group, attempts: int) -> dict:
        """恒常失敗個体の救済。試行ごとに seed 塩と要素組合せを変えて N 回まで引く。

        seed 列 = `blake2b(seed, pid, "backstory", "rescue<k>")` の純関数 = 再実行で同じ列。
        """
        pid = str(rec["id"])
        tokens = 0
        reason = "unknown"
        sha = ""
        for a in range(1, int(attempts) + 1):
            system, user, elements = build_prompt(rec, int(self.cfg.seed), attempt=a)
            sha = prompt_sha(system, user)
            seed = stable_seed(int(self.cfg.seed), pid, "backstory", f"rescue{a}")
            raw, tok = self._call(group, pid, system, user, seed)
            tokens += tok
            if raw.startswith("__error__"):
                reason = raw[len("__error__"):]
                continue
            text = normalize(raw)
            reason = check(text) or ""
            if not reason:
                line = {"pid": pid, "backstory": text, "model": group.model,
                        "seed": seed, "prompt_sha": sha,
                        "elements": list(elements), "rescue": a}
                return {"ok": True, "line": line, "tokens": tokens,
                        "attempts": a, "retried": True}
        return {"ok": False, "pid": pid, "reason": reason or "unknown",
                "prompt_sha": sha, "tokens": tokens,
                "attempts": int(attempts), "retried": True}


# ---------------------------------------------------------------- 入出力
def load_only_ids(path: Path | None) -> set[str] | None:
    """生成対象を絞る pid 集合(None = 全件)。

    受け付ける形: ``{"ids": [...]}`` / ``[...]`` の JSON、または 1 行 1 pid のテキスト。
    ★プールに同梱の `llm_targets.json`(LLM 発話対象の id 集合)をそのまま渡せる。
    """
    if path is None:
        return None
    raw = Path(path).read_text(encoding="utf-8").strip()
    if raw.startswith(("{", "[")):
        data = json.loads(raw)
        ids = data.get("ids", []) if isinstance(data, dict) else data
        return {str(x) for x in ids}
    return {ln.strip() for ln in raw.splitlines() if ln.strip()}


def iter_pool(pool_dir: Path, layer: str):
    """<pool>/<layer>/part-*.jsonl を**逐次**読む(100 万行を丸ごと持たない)。"""
    layer_dir = pool_dir / layer
    for part in sorted(layer_dir.glob("part-*.jsonl")):
        with part.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _repair(path: Path, keep: int) -> None:
    """壊れた尾を落として書き直す(読めた `keep` 行だけを残す)。"""
    tmp = path.with_suffix(".gz.tmp")
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as src, \
            gzip.open(tmp, "wt", encoding="utf-8") as dst:
        try:
            for line in src:
                if n >= keep:
                    break
                dst.write(line)
                n += 1
        except (EOFError, OSError, zlib.error):
            pass
    tmp.replace(path)


def load_done(path: Path) -> tuple[set[str], set[int], int]:
    """既存出力を読み、(済み pid, 重複ハッシュ集合, 重複件数)。壊れていれば書き直す。"""
    done: set[str] = set()
    seen: set[int] = set()
    dups = 0
    if not path.exists():
        return done, seen, dups
    good = 0
    truncated = False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    pid = str(rec["pid"])
                except Exception:                                  # noqa: BLE001
                    truncated = True
                    break
                done.add(pid)
                key = dup_key(str(rec.get("backstory", "")))
                if key in seen:
                    dups += 1
                else:
                    seen.add(key)
                good += 1
    except (EOFError, OSError, zlib.error):
        truncated = True
    if truncated:
        print(f"[bs] 壊れた末尾を検出: {path.name} を {good} 行へ復旧",
              file=sys.stderr, flush=True)
        _repair(path, good)
    return done, seen, dups


class _Appender:
    """最初の 1 行を書くまでファイルを開かない追記器。

    gzip を追記モードで開くと、1 行も書かなくても**空メンバー**が足されてファイルが
    育つ(= resume が冪等でなくなる)。書くものが在るときだけ開くことでそれを避ける。
    """

    def __init__(self, path: Path, *, gz: bool, compresslevel: int = 6):
        self.path = path
        self.gz = gz
        self.compresslevel = int(compresslevel)
        self._fh = None

    def write(self, line: str) -> None:
        if self._fh is None:
            self._fh = (gzip.open(self.path, "at", encoding="utf-8",
                                  compresslevel=self.compresslevel)
                        if self.gz else self.path.open("a", encoding="utf-8"))
        self._fh.write(line)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class Progress:
    """10 秒ごとに done/total・速度・ETA を stderr へ出す(スレッド 1 本)。"""

    def __init__(self, total: int, every_s: float):
        self.total = total
        self.every_s = float(every_s)
        self.lock = threading.Lock()
        self.n = 0
        self.ok = 0
        self.fail = 0
        self.retry = 0
        self.tokens = 0
        self.skipped = 0
        self.label = ""
        self.t0 = time.time()
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def start(self, label: str) -> None:
        self.label = label
        if self.every_s > 0:
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self.emit()

    def bump(self, *, ok: bool, tokens: int, retried: bool) -> None:
        with self.lock:
            self.n += 1
            self.tokens += tokens
            self.ok += 1 if ok else 0
            self.fail += 0 if ok else 1
            self.retry += 1 if retried else 0

    def emit(self) -> None:
        with self.lock:
            n, ok, fail, retry, tok = self.n, self.ok, self.fail, self.retry, self.tokens
        el = max(1e-6, time.time() - self.t0)
        rate = n / el
        rest = max(0, self.total - n)
        eta = rest / rate if rate > 0 else 0.0
        print(f"[bs] {self.label} {n}/{self.total} ok={ok} retry={retry} fail={fail} "
              f"skip={self.skipped} {rate:.1f}/s tok/s={tok / el:.0f} "
              f"ETA={eta / 3600:.2f}h", file=sys.stderr, flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)


def run_layer(layer: str, group: Group, cfg: argparse.Namespace,
              out_dir: Path, total_hint: int) -> dict:
    """1 層を生成して <out>/<layer>.jsonl.gz へ追記する。戻り値 = 層の統計。"""
    out_path = out_dir / f"{layer}.jsonl.gz"
    fail_path = out_dir / f"{layer}.failed.jsonl"
    done, seen, dups = load_done(out_path)
    only_ids = getattr(cfg, "only_ids_set", None)
    if only_ids is not None:
        # 進捗の分母も対象集合に合わせる。★pid は "<層><接尾?>_<連番>"(実測の接尾:
        # L3reg_ = 定期来街者・L5c_ = 議員)なので、層は**接頭辞一致**で見る。
        total_hint = sum(1 for x in only_ids if x.startswith(layer))
    limit = int(cfg.limit) if int(cfg.limit) > 0 else 0
    # --limit は「層の先頭 N レコード」を意味する(resume しても同じ N を見る)。
    if limit and total_hint > 0:
        target = min(total_hint, limit)
    elif limit:
        target = limit
    else:
        target = total_hint
    prog = Progress(max(0, target - len(done)), cfg.progress_s)
    prog.skipped = len(done)
    prog.start(f"{layer}({group.name}/{group.model})")
    runner = Runner(cfg)
    lengths: list[int] = []
    written = 0
    failed = 0
    t0 = time.time()
    inflight: set = set()
    bound = max(2, int(cfg.concurrency) * 2)

    def drain(fut_set, fh, ffh) -> None:
        nonlocal written, failed, dups, consec_fail
        for fut in fut_set:
            res = fut.result()
            prog.bump(ok=res["ok"], tokens=res["tokens"], retried=res["retried"])
            if res["ok"]:
                consec_fail = 0
                line = res["line"]
                key = dup_key(line["backstory"])
                if key in seen:
                    dups += 1
                else:
                    seen.add(key)
                lengths.append(len(line["backstory"]))
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
                if written % int(cfg.flush_every) == 0:
                    fh.flush()
            else:
                ffh.write(json.dumps({"pid": res["pid"], "reason": res["reason"],
                                      "prompt_sha": res["prompt_sha"]},
                                     ensure_ascii=False) + "\n")
                ffh.flush()
                failed += 1
                consec_fail += 1
                if abort_after and consec_fail >= abort_after:
                    raise _AbortRun(
                        f"{layer}: 連続 {consec_fail} 件失敗(直近の理由 "
                        f"{res['reason']})= サーバ/モデルを確認して再実行")

    seen_records = 0
    consec_fail = 0
    abort_after = int(cfg.abort_after_failures)
    fh = _Appender(out_path, gz=True, compresslevel=int(cfg.compresslevel))
    ffh = _Appender(fail_path, gz=False)
    try:
        with ThreadPoolExecutor(max_workers=int(cfg.concurrency)) as pool:
            for rec in iter_pool(cfg.pool, layer):
                pid = str(rec.get("id"))
                if only_ids is not None and pid not in only_ids:
                    continue                       # 対象外(--only-ids)
                if limit and seen_records >= limit:
                    break
                seen_records += 1
                if pid in done:
                    continue
                inflight.add(pool.submit(runner.run, rec, group))
                if len(inflight) >= bound:
                    ready, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    drain(ready, fh, ffh)
            while inflight:
                ready, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                drain(ready, fh, ffh)
    finally:
        prog.stop()
        fh.close()
        ffh.close()
    prog.emit()
    elapsed = time.time() - t0
    lengths.sort()
    return {
        "layer": layer,
        "group": group.name,
        "model": group.model,
        "records_scanned": seen_records,
        "skipped_existing": len(done),
        "written": written,
        "failed": failed,
        "retried": prog.retry,
        "completion_tokens": prog.tokens,
        "elapsed_s": round(elapsed, 1),
        "rate_per_s": round(prog.n / elapsed, 2) if elapsed > 0 else 0.0,
        "length_chars": ({
            "min": lengths[0], "p50": lengths[len(lengths) // 2],
            "p95": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))],
            "max": lengths[-1],
            "mean": round(sum(lengths) / len(lengths), 1),
        } if lengths else {}),
        "duplicate_lines": dups,
        "unique_backstories": len(seen),
        "duplicate_rate": round(dups / max(1, len(seen) + dups), 6),
    }


def _read_failed(path: Path) -> tuple[dict, list]:
    """failed.jsonl → (pid → 最後の行, 読めなかった生行)。pid が同じ行は後勝ちで畳む。"""
    rows: dict = {}
    broken: list = []
    if not path.exists():
        return rows, broken
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
            pid = str(rec.get("pid", "") or "")
        except ValueError:
            broken.append(raw)
            continue
        if pid:
            rows[pid] = rec
        else:
            broken.append(raw)
    return rows, broken


def rescue_layer(layer: str, group: Group, cfg: argparse.Namespace,
                 out_dir: Path) -> dict | None:
    """`<layer>.failed.jsonl` に居る pid だけを、試行ごとに条件を変えて拾い直す。

    * 成功 → 本体 `<layer>.jsonl.gz` へ追記し、failed から**外す**。
    * 全 attempt 失敗 → failed に残す(理由と試行回数を更新)。
    * 既に本体へ入っている pid(前回の救済で回収済み)は撃たずに failed から外すだけ。
    None を返すのは「その層に failed が無い = 何もしない」場合。
    """
    out_path = out_dir / f"{layer}.jsonl.gz"
    fail_path = out_dir / f"{layer}.failed.jsonl"
    rows, broken = _read_failed(fail_path)
    if not rows and not broken:
        return None
    attempts = int(cfg.rescue)
    done, seen, dups = load_done(out_path)
    already = sorted(pid for pid in rows if pid in done)     # 既に本体に在る
    pending = sorted(pid for pid in rows if pid not in done)
    limit = int(cfg.limit) if int(cfg.limit) > 0 else 0
    if limit:
        pending = pending[:limit]
    targets = set(pending)
    # プールを 1 度だけ流して対象 record を拾う(救済対象は少数=保持して良い)
    recs = [rec for rec in iter_pool(cfg.pool, layer)
            if str(rec.get("id")) in targets]
    found = {str(r["id"]) for r in recs}

    prog = Progress(len(recs), cfg.progress_s)
    prog.skipped = len(already)
    prog.start(f"{layer}(rescue x{attempts}/{group.name}/{group.model})")
    runner = Runner(cfg)
    lengths: list[int] = []
    rescued: set[str] = set()
    still: dict = {}
    t0 = time.time()
    inflight: set = set()
    bound = max(2, int(cfg.concurrency) * 2)

    def drain(fut_set, fh) -> None:
        nonlocal dups
        for fut in fut_set:
            res = fut.result()
            prog.bump(ok=res["ok"], tokens=res["tokens"], retried=True)
            if res["ok"]:
                line = res["line"]
                key = dup_key(line["backstory"])
                if key in seen:
                    dups += 1
                else:
                    seen.add(key)
                lengths.append(len(line["backstory"]))
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                fh.flush()
                rescued.add(line["pid"])
            else:
                base = dict(rows.get(res["pid"], {"pid": res["pid"]}))
                base.update({"reason": res["reason"],
                             "prompt_sha": res["prompt_sha"],
                             "rescue_attempts": res["attempts"]})
                still[res["pid"]] = base

    fh = _Appender(out_path, gz=True, compresslevel=int(cfg.compresslevel))
    try:
        with ThreadPoolExecutor(max_workers=int(cfg.concurrency)) as pool:
            for rec in recs:
                inflight.add(pool.submit(runner.run_rescue, rec, group, attempts))
                if len(inflight) >= bound:
                    ready, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    drain(ready, fh)
            while inflight:
                ready, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                drain(ready, fh)
    finally:
        prog.stop()
        fh.close()
    prog.emit()

    # failed.jsonl を書き直す。残すのは (a) 全 attempt 落ちた個体 (b) 今回撃たなかった個体
    #   = プールに居ない pid / --limit で後回しにした pid / 壊れた生行。
    keep: list[str] = []
    for pid, rec in rows.items():
        if pid in rescued or pid in already:
            continue                                   # 回収済み = 落とす
        if pid in still:
            keep.append(json.dumps(still[pid], ensure_ascii=False))
        elif pid not in found and pid in targets:      # プールに居ない = 正直に残す
            row = dict(rec)
            row["reason"] = "not_in_pool"
            keep.append(json.dumps(row, ensure_ascii=False))
        else:
            keep.append(json.dumps(rec, ensure_ascii=False))
    keep += broken
    tmp = fail_path.with_suffix(".jsonl.tmp")
    if keep:
        tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
        tmp.replace(fail_path)
    else:                                              # 全部回収した = 空ファイルを残さない
        if tmp.exists():
            tmp.unlink()
        fail_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    lengths.sort()
    return {
        "layer": layer,
        "group": group.name,
        "model": group.model,
        "mode": "rescue",
        "rescue_attempts": attempts,
        "targets": len(rows),
        "attempted": len(recs),
        "rescued": len(rescued),
        "remaining": len(keep),
        "already_in_body": len(already),
        "skipped_existing": len(already),
        "written": len(rescued),
        "failed": len(still),
        "retried": prog.retry,
        "completion_tokens": prog.tokens,
        "elapsed_s": round(elapsed, 1),
        "rate_per_s": round(prog.n / elapsed, 2) if elapsed > 0 else 0.0,
        "length_chars": ({
            "min": lengths[0], "p50": lengths[len(lengths) // 2],
            "p95": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))],
            "max": lengths[-1],
            "mean": round(sum(lengths) / len(lengths), 1),
        } if lengths else {}),
        "duplicate_lines": dups,
        "unique_backstories": len(seen),
        "duplicate_rate": round(dups / max(1, len(seen) + dups), 6),
    }


# ---------------------------------------------------------------- CLI
def _servers(text: str) -> list[str]:
    return [s.strip().rstrip("/") for s in str(text).split(",") if s.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, default=Path("data/persona_pool_v2"))
    p.add_argument("--out", type=Path, default=Path("data/persona_backstory_v2"))
    p.add_argument("--layers", default=",".join(_LAYERS),
                   help="処理する層の包含フィルタ。例 L2,L4,L5(既定=全層)")
    p.add_argument("--core-layers", default="L1,L3",
                   help="core グループへ回す層(残りは mass)")
    p.add_argument("--core-servers", default="http://localhost:8005,http://localhost:8006")
    # ★モデル名はスクリプトに固定しない(起動側の served name をそのまま送る)。
    #   既定は空 = 「この段では使わない」の意思表示。使う層があるのに空なら起動時エラー。
    p.add_argument("--core-model", default="",
                   help="core サーバの served name(起動側の宣言をそのまま送る)")
    p.add_argument("--mass-servers",
                   default=("http://localhost:8000,http://localhost:8001,"
                            "http://localhost:8002,http://localhost:8003,"
                            "http://localhost:8004"))
    p.add_argument("--mass-model", default="",
                   help="mass サーバの served name(既定=空 = この段では使わない)")
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="層ごとの上限件数(0=全件)")
    p.add_argument("--rescue", type=int, default=0, metavar="N",
                   help=("救済モード: failed.jsonl の pid だけを、試行ごとに seed 塩と"
                         "要素組合せを変えて N 回まで引き直す(0=通常生成)"))
    p.add_argument("--only-ids", type=Path, default=None,
                   help=("この pid 集合だけを生成(JSON {\"ids\":[...]} / 配列 / 1 行 1 pid)。"
                         "例: プール同梱の llm_targets.json"))
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--http-retries", type=int, default=2)
    p.add_argument("--progress-s", type=float, default=10.0)
    p.add_argument("--flush-every", type=int, default=200)
    p.add_argument("--abort-after-failures", type=int, default=500,
                   help="連続失敗がこの数に達したら中止(0=無効)。無人運転の保険")
    p.add_argument("--compresslevel", type=int, default=6)
    p.add_argument("--no-probe", action="store_true", help="/v1/models の来歴取得をしない")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    core = Group("core", _servers(cfg.core_servers), cfg.core_model)
    mass = Group("mass", _servers(cfg.mass_servers), cfg.mass_model)
    layers = [x.strip() for x in str(cfg.layers).split(",") if x.strip()]
    core_layers = {x.strip() for x in str(cfg.core_layers).split(",") if x.strip()}
    # ---- 片側運転(2 段運用)------------------------------------------------------
    # このランで**実際に走る層**だけを見て、要るグループの設定だけを要求する。
    # 例: `--layers L2,L4,L5`(全部 mass)なら core の servers/model は未指定で良い。
    # 逆に、層がルーティングされるのに設定が無いグループは**黙って動かさず**止める。
    present = [x for x in layers if (Path(cfg.pool) / x).is_dir()]
    group_layers = {
        "core": [x for x in layers if x in core_layers],
        "mass": [x for x in layers if x not in core_layers],
    }
    used = {"core": [x for x in present if x in core_layers],
            "mass": [x for x in present if x not in core_layers]}
    for g in (core, mass):
        if not used[g.name]:
            continue                          # この段では使わない = 未指定でも構わない
        missing = [f"--{g.name}-servers" for _ in (1,) if not g.servers]
        missing += [f"--{g.name}-model" for _ in (1,) if not str(g.model).strip()]
        if missing:
            raise SystemExit(
                f"{g.name} グループへ回す層 {used[g.name]} があるのに "
                f"{' / '.join(missing)} が未指定です"
                f"(この段で走らせないなら --layers か --core-layers で外す)")
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.only_ids_set = load_only_ids(cfg.only_ids)

    pool_meta_path = Path(cfg.pool) / "meta.json"
    pool_counts: dict = {}
    if pool_meta_path.exists():
        pool_counts = json.loads(pool_meta_path.read_text(encoding="utf-8")).get(
            "layer_counts", {}) or {}

    meta: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/build_persona_backstory.py",
        "prompt_version": PROMPT_VERSION,
        "prompt_system_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16],
        "elements": [e[0] for e in ELEMENTS],
        "pool": str(cfg.pool),
        "seed": int(cfg.seed),
        "temperature": float(cfg.temperature),
        "max_tokens": int(cfg.max_tokens),
        "top_p": "unset(サーバ既定)",
        "api_mode": "chat",
        "enable_thinking": False,
        "length_band_chars": [LEN_MIN, LEN_MAX],
        "length_target_chars": [TARGET_MIN, TARGET_MAX],
        "concurrency": int(cfg.concurrency),
        "mode": ("rescue" if int(cfg.rescue) > 0 else "generate"),
        "rescue_attempts": int(cfg.rescue),
        "only_ids": (None if cfg.only_ids is None else
                     {"path": str(cfg.only_ids), "count": len(cfg.only_ids_set or ())}),
        "groups": {
            g.name: {"model": g.model, "served_name": g.model,
                     "servers": g.servers, "layers": group_layers[g.name],
                     "used": bool(used[g.name])}
            for g in (core, mass)
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "layers": [],
    }
    if not cfg.no_probe:                      # プローブは**この段で使う群だけ**
        for g in (core, mass):
            if used[g.name]:
                meta["groups"][g.name]["probe"] = probe_models(g.servers)

    # 2 段運用 / resume で meta.json を上書きしても**前段の来歴を失わない**
    # (run_manifest と同じ流儀: 前回分を history へ畳む。直近 10 段まで)。
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            prev = None
        if isinstance(prev, dict):
            hist = prev.pop("previous_stages", [])
            meta["previous_stages"] = ([x for x in hist if isinstance(x, dict)]
                                       + [prev])[-10:]
    try:
        for layer in layers:
            if not (Path(cfg.pool) / layer).is_dir():
                print(f"[bs] {layer}: プールに層が無い→スキップ", file=sys.stderr)
                continue
            group = core if layer in core_layers else mass
            if int(cfg.rescue) > 0:            # 救済モード = failed.jsonl の pid だけ
                stat = rescue_layer(layer, group, cfg, out_dir)
                if stat is None:
                    print(f"[bs] {layer}: failed.jsonl が無い→スキップ", file=sys.stderr)
                    continue
            else:
                total = int(pool_counts.get(layer, 0) or 0)
                stat = run_layer(layer, group, cfg, out_dir, total)
            meta["layers"].append(stat)
            meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    except KeyboardInterrupt:
        meta["interrupted"] = True
        meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print("[bs] 中断(resume 可能: 同じコマンドで続きから)", file=sys.stderr)
        return 130
    except _AbortRun as exc:
        meta["aborted"] = str(exc)
        meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"[bs] 中止: {exc}(resume 可能)", file=sys.stderr)
        return 2
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["totals"] = {
        "written": sum(x["written"] for x in meta["layers"]),
        "failed": sum(x["failed"] for x in meta["layers"]),
        "retried": sum(x["retried"] for x in meta["layers"]),
        "skipped_existing": sum(x["skipped_existing"] for x in meta["layers"]),
        "completion_tokens": sum(x["completion_tokens"] for x in meta["layers"]),
        "elapsed_s": round(sum(x["elapsed_s"] for x in meta["layers"]), 1),
    }
    if int(cfg.rescue) > 0:                   # 救済の収支(対象/回収/残)
        meta["rescue"] = {
            "attempts": int(cfg.rescue),
            "targets": sum(x.get("targets", 0) for x in meta["layers"]),
            "attempted": sum(x.get("attempted", 0) for x in meta["layers"]),
            "rescued": sum(x.get("rescued", 0) for x in meta["layers"]),
            "remaining": sum(x.get("remaining", 0) for x in meta["layers"]),
        }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    if int(cfg.rescue) > 0:
        r = meta["rescue"]
        print(f"[bs] 救済完了 対象={r['targets']} 回収={r['rescued']} "
              f"残={r['remaining']} → {out_dir}", file=sys.stderr)
        return 0
    print(f"[bs] 完了 written={meta['totals']['written']} "
          f"failed={meta['totals']['failed']} → {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
