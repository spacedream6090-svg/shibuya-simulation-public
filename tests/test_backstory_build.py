"""ペルソナ来歴の事前生成(scripts/build_persona_backstory.py)の検収テスト。

実 LLM も GPU も使わない。OpenAI 互換の最小スタブ HTTP サーバ(tests/test_fleet.py と
同じ流儀)を立て、
  (a) プロンプト組立の決定論(同 pid → 同 prompt_sha)と要素選択の分布
  (b) core / mass 2 グループへの層割当と sticky 割当
  (c) 送出ボディの形(top_p を送らない・seed を送る)
  (d) 検品(空 / JSON 混入 / 英語混入 / 長さ帯外の 1 回だけ再生成)
  (e) resumable(出力済み pid はスキップ・壊れた gz の末尾は復旧)
  (f) 出力スキーマ・meta.json・プール本体不触
を確認する。
"""
from __future__ import annotations

import gzip
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_persona_backstory as bb  # noqa: E402

# 検品を通る見本(94 字)。末尾のかなだけ差し替えて重複を避ける。
_GOOD = ("幼い頃は海沿いの町で過ごし、祖母の家から高校へ通っていた。上京してからは"
         "職場と住まいを往復する日々が続いている。最近は商店街の移り変わりが気になっている")
_KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねの"


def _good_text(n: int) -> str:
    """スタブの正常応答(呼ぶたび末尾が変わる = 重複率テストと切り分けられる)。"""
    return _GOOD + _KANA[n % len(_KANA)] + "。"


# ---------------------------------------------------------------- スタブ
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):            # テスト出力を汚さない
        return

    def _send(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):                        # /v1/models(来歴プローブ)
        self._send({"data": [{"id": self.server.tag, "root": self.server.tag,
                              "max_model_len": 8192}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            req = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            req = {}
        srv = self.server
        with srv.lock:
            srv.hits.append(req)
            idx = len(srv.hits)
        text = srv.responder(req, idx)
        self._send({"choices": [{"message": {"content": text}}],
                    "usage": {"completion_tokens": 90}})


def _start_stub(tag: str, responder=None):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.hits = []
    srv.lock = threading.Lock()
    srv.tag = tag
    srv.responder = responder or (lambda req, i: _good_text(i))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ---------------------------------------------------------------- 小さなプール
def _rec(pid: str, layer: str, **over) -> dict:
    base = {
        "id": pid, "layer": layer, "presence": "resident",
        "name": "架空 太郎", "age": 34, "gender": "男", "occupation": "会社員",
        "industry_major": "情報通信業", "role": "一般", "employment": "正規",
        "rank": "一般", "household_type": "単身", "household_role": "世帯主",
        "persona": "あなたは架空太郎、34歳の会社員(男性)。渋谷の街に住んでいる。"
                   "散歩が好きで、丁寧な物腰。自分の言葉で自然に、短く話す。",
    }
    base.update(over)
    return base


@pytest.fixture()
def pool(tmp_path: Path) -> Path:
    """L1(6 人)・L2(8 人)だけの極小プール。"""
    root = tmp_path / "pool"
    counts = {"L1": 6, "L2": 8}
    for layer, n in counts.items():
        d = root / layer
        d.mkdir(parents=True)
        lines = [json.dumps(_rec(f"{layer}_{i:08d}", layer), ensure_ascii=False)
                 for i in range(n)]
        (d / "part-0000.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "meta.json").write_text(json.dumps({"layer_counts": counts}),
                                    encoding="utf-8")
    return root


def _run(pool_dir: Path, out: Path, core_url: str, mass_url: str, *extra) -> int:
    argv = ["--pool", str(pool_dir), "--out", str(out),
            "--core-servers", core_url, "--core-model", "stub-core",
            "--mass-servers", mass_url, "--mass-model", "stub-mass",
            "--core-layers", "L1", "--layers", "L1,L2", "--concurrency", "4",
            "--progress-s", "0", "--http-retries", "0", "--no-probe", *extra]
    return bb.main([str(a) for a in argv])


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


# ---------------------------------------------------------------- (a) 決定論
def test_prompt_is_deterministic_per_pid():
    """同じ record・同じ seed なら prompt も prompt_sha も 1 バイト変わらない。"""
    rec = _rec("L1_00000000", "L1")
    s1, u1, e1 = bb.build_prompt(rec, 42)
    s2, u2, e2 = bb.build_prompt(dict(rec), 42)
    assert (s1, u1, e1) == (s2, u2, e2)
    assert bb.prompt_sha(s1, u1) == bb.prompt_sha(s2, u2)
    other = bb.build_prompt(_rec("L1_00000001", "L1", name="架空 次郎"), 42)
    assert bb.prompt_sha(*other[:2]) != bb.prompt_sha(s1, u1)
    assert len(bb.prompt_sha(s1, u1)) == 16


def test_elements_are_two_distinct_and_cover_all_five():
    """要素は pid ハッシュで相異なる 2 つ。500 人で 5 要素すべてが使われる。"""
    used: dict[str, int] = {}
    for i in range(500):
        a, b = bb.pick_elements(f"L4_{i:08d}", 42)
        assert a != b
        for k in (a, b):
            used[bb.ELEMENTS[k][0]] = used.get(bb.ELEMENTS[k][0], 0) + 1
    assert set(used) == {e[0] for e in bb.ELEMENTS}
    assert min(used.values()) > 500 * 2 * 0.05      # 極端な偏りが無い


def test_request_seed_stable_and_retry_differs():
    """seed は (seed, pid, 用途, retry) の安定値。retry で必ず変わる・31bit に収まる。"""
    s = bb.stable_seed(42, "L1_00000000")
    assert s == bb.stable_seed(42, "L1_00000000")
    assert 0 <= s <= 0x7FFFFFFF
    assert bb.stable_seed(42, "L1_00000000", "backstory", "retry") != s
    assert bb.stable_seed(7, "L1_00000000") != s


def test_sticky_server_assignment():
    """同じ pid は常に同じサーバへ(prefix cache)。全サーバが使われる。"""
    servers = [f"http://localhost:{8000 + i}" for i in range(5)]
    pid = "L2_00001234"
    assert bb.pick_server(pid, servers) == bb.pick_server(pid, servers)
    hit = {bb.pick_server(f"L2_{i:08d}", servers) for i in range(300)}
    assert hit == set(servers)


def test_prompt_carries_skeleton_constraints():
    """骨格は**制約**として本文に載る。無い欄は書かない(捏造しない)。"""
    child = _rec("L1_00000009", "L1", age=0, occupation="未就学児",
                 school_stage="保育所", industry_major="")
    _, user, _ = bb.build_prompt(child, 42)
    assert "通学先: 保育所" in user and "業種" not in user
    assert "0歳" in user
    visitor = _rec("L4_00000001", "L4", presence="stochastic", is_foreign=True,
                   visit_purpose="買い物")
    _, user2, _ = bb.build_prompt(visitor, 42)
    assert "訪日外国人" in user2 and "来街目的: 買い物" in user2
    assert "実在" in bb.SYSTEM_PROMPT and "矛盾" in bb.SYSTEM_PROMPT


# ---------------------------------------------------------------- (d) 検品
def test_normalize_strips_think_label_and_tail():
    """思考ブロック・先頭ラベル・改行・尻切れ文を落とす。"""
    raw = "<think>ここは思考</think>\n【来歴】" + _GOOD + "。\n途中で切れた文"
    out = bb.normalize(raw)
    assert out.startswith("幼い頃は") and out.endswith("。")
    assert "think" not in out and "【" not in out and "\n" not in out
    assert "途中で切れた文" not in out
    assert bb.normalize("") == ""


@pytest.mark.parametrize("text,reason", [
    ("", "empty"),
    ('{"backstory": "あああ"}', "json"),
    ("He grew up near the sea and later moved to the city for a new job there.", "english"),
    ("短い経歴。", "short"),
    (_GOOD * 3, "long"),
])
def test_check_rejects_bad_text(text, reason):
    assert bb.check(text) == reason
    assert bb.check(_good_text(0)) is None


def test_check_rejects_non_japanese_charset():
    """キリル文字などの混入は charset で落とす。"""
    cyr = "".join(chr(c) for c in (0x0434, 0x043E, 0x043C))   # キリル文字 3 字
    assert bb.check(_GOOD + "。" + cyr) == "charset"


# ---------------------------------------------------------------- (b)(c)(f) e2e
def test_end_to_end_schema_and_routing(pool: Path, tmp_path: Path):
    """L1=core / L2=mass へ割り当てて生成し、出力スキーマと model を検査する。"""
    core, core_url = _start_stub("stub-core")
    mass, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    assert _run(pool, out, core_url, mass_url) == 0

    l1, l2 = _lines(out / "L1.jsonl.gz"), _lines(out / "L2.jsonl.gz")
    assert len(l1) == 6 and len(l2) == 8
    assert len(core.hits) == 6 and len(mass.hits) == 8      # 層割当が守られている
    for line in l1 + l2:
        assert set(line) <= {"pid", "backstory", "model", "seed", "prompt_sha",
                             "elements", "retry", "rescue"}
        assert {"pid", "backstory", "model", "seed", "prompt_sha"} <= set(line)
        assert bb.check(line["backstory"]) is None
        assert len(line["elements"]) == 2
    assert {x["model"] for x in l1} == {"stub-core"}
    assert {x["model"] for x in l2} == {"stub-mass"}
    rec = json.loads((pool / "L1" / "part-0000.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    ref = bb.build_prompt(rec, 42)
    first = [x for x in l1 if x["pid"] == rec["id"]][0]
    assert first["prompt_sha"] == bb.prompt_sha(*ref[:2])
    assert first["seed"] == bb.stable_seed(42, rec["id"])


def test_request_body_shape(pool: Path, tmp_path: Path):
    """送出ボディ: top_p を送らない・seed / temperature / max_tokens / chat 経路。"""
    core, core_url = _start_stub("stub-core")
    mass, mass_url = _start_stub("stub-mass")
    _run(pool, tmp_path / "out", core_url, mass_url, "--temperature", "0.9",
         "--max-tokens", "200")
    body = core.hits[0]
    assert "top_p" not in body and "top_k" not in body
    assert body["temperature"] == 0.9 and body["max_tokens"] == 200
    assert body["model"] == "stub-core"
    assert isinstance(body["seed"], int) and 0 <= body["seed"] <= 0x7FFFFFFF
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["stream"] is False


def test_pool_files_are_never_touched(pool: Path, tmp_path: Path):
    """プール本体は 1 バイトも変わらない(読むだけ)。"""
    before = {p: p.read_bytes() for p in pool.rglob("*") if p.is_file()}
    _, core_url = _start_stub("stub-core")
    _, mass_url = _start_stub("stub-mass")
    _run(pool, tmp_path / "out", core_url, mass_url)
    after = {p: p.read_bytes() for p in pool.rglob("*") if p.is_file()}
    assert before == after


# ---------------------------------------------------------------- (d) 再生成
def test_short_response_is_regenerated_once(pool: Path, tmp_path: Path):
    """長さ帯を外したら retry seed で 1 度だけ引き直す(2 回目で合格 → 保存)。"""
    # retry seed のときだけ合格文を返す = 全件がちょうど 1 度だけ引き直される
    retry_seeds = {bb.stable_seed(42, f"L1_{i:08d}", "backstory", "retry")
                   for i in range(6)}

    def responder(req, i):
        return _good_text(i) if req["seed"] in retry_seeds else "短い経歴。"

    core, core_url = _start_stub("stub-core", responder)
    _, mass_url = _start_stub("stub-mass", responder)
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url, "--layers", "L1")
    lines = _lines(out / "L1.jsonl.gz")
    assert len(lines) == 6 and len(core.hits) == 12
    for line in lines:
        assert bb.check(line["backstory"]) is None
        assert line["retry"] == 1
        assert line["seed"] == bb.stable_seed(42, line["pid"], "backstory", "retry")


def test_bad_response_is_not_written_but_recorded(pool: Path, tmp_path: Path):
    """空応答は 2 回引いても落ちる → 本体には書かず failed.jsonl に理由を残す。"""
    core, core_url = _start_stub("stub-core", lambda req, i: "")
    _, mass_url = _start_stub("stub-mass", lambda req, i: "")
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url, "--layers", "L1")
    assert _lines(out / "L1.jsonl.gz") == []
    failed = [json.loads(x) for x in
              (out / "L1.failed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(failed) == 6
    assert {x["reason"] for x in failed} == {"empty"}
    assert len(core.hits) == 12                      # 6 人 × (本番 + 再生成)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["totals"]["failed"] == 6 and meta["totals"]["written"] == 0


def test_aborts_after_consecutive_failures(pool: Path, tmp_path: Path):
    """連続失敗が閾値に達したら中止(rc=2)。無人運転で何十万件も空振りさせない。"""
    _, core_url = _start_stub("stub-core", lambda req, i: "")
    _, mass_url = _start_stub("stub-mass", lambda req, i: "")
    out = tmp_path / "out"
    rc = _run(pool, out, core_url, mass_url, "--layers", "L1",
              "--concurrency", "1", "--abort-after-failures", "2")
    assert rc == 2
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert "aborted" in meta and "L1" in meta["aborted"]
    assert meta["layers"] == []


def test_english_and_json_contamination_rejected(pool: Path, tmp_path: Path):
    """英語・JSON 混入は保存しない(理由が failed.jsonl に残る)。"""
    _, core_url = _start_stub(
        "stub-core", lambda req, i: "He moved to the city and started a new job there now.")
    _, mass_url = _start_stub(
        "stub-mass", lambda req, i: '{"backstory": "' + _GOOD + '。"}')
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url)
    r1 = {json.loads(x)["reason"] for x in
          (out / "L1.failed.jsonl").read_text(encoding="utf-8").splitlines()}
    r2 = {json.loads(x)["reason"] for x in
          (out / "L2.failed.jsonl").read_text(encoding="utf-8").splitlines()}
    assert r1 == {"english"} and r2 == {"json"}


# ---------------------------------------------------------------- (e) resume
def test_resume_skips_existing_pids(pool: Path, tmp_path: Path):
    """2 回目は 1 リクエストも投げない(出力済み pid は読み飛ばす)。"""
    core, core_url = _start_stub("stub-core")
    mass, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url)
    n_first = len(core.hits) + len(mass.hits)
    before = (out / "L1.jsonl.gz").read_bytes()
    _run(pool, out, core_url, mass_url)
    assert len(core.hits) + len(mass.hits) == n_first == 14
    assert (out / "L1.jsonl.gz").read_bytes() == before
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["totals"]["skipped_existing"] == 14 and meta["totals"]["written"] == 0


def test_truncated_gz_tail_is_repaired(pool: Path, tmp_path: Path):
    """異常終了で壊れた gz の末尾は、読めた行まで書き直してから追記を再開する。"""
    core, core_url = _start_stub("stub-core")
    mass, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url, "--layers", "L1")
    path = out / "L1.jsonl.gz"
    good = _lines(path)
    path.write_bytes(path.read_bytes() + b"\x1f\x8b\x08\x00broken-tail")
    done, seen, dups = bb.load_done(path)
    assert len(done) == len(good) == 6
    assert _lines(path) == good                     # 復旧後も中身は同じ
    _run(pool, out, core_url, mass_url, "--layers", "L1")
    assert len(_lines(path)) == 6                   # 二重書きしない


def test_only_ids_restricts_generation(pool: Path, tmp_path: Path):
    """--only-ids で対象を絞れる(プール同梱 llm_targets.json をそのまま渡せる形)。"""
    ids = tmp_path / "targets.json"
    ids.write_text(json.dumps({"ids": ["L1_00000001", "L1_00000004", "L2_00000003"]}),
                   encoding="utf-8")
    core, core_url = _start_stub("stub-core")
    mass, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url, "--only-ids", str(ids))
    # 行の並びは**完了順**(並列生成なので pool 順ではない)= 集合で見る
    assert sorted(x["pid"] for x in _lines(out / "L1.jsonl.gz")) == ["L1_00000001",
                                                                    "L1_00000004"]
    assert [x["pid"] for x in _lines(out / "L2.jsonl.gz")] == ["L2_00000003"]
    assert len(core.hits) == 2 and len(mass.hits) == 1
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["only_ids"]["count"] == 3


def test_limit_caps_records_per_layer(pool: Path, tmp_path: Path):
    """--limit は層の先頭 N 件だけを見る(疎通確認用)。"""
    core, core_url = _start_stub("stub-core")
    _, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url, "--layers", "L1", "--limit", "2")
    assert len(_lines(out / "L1.jsonl.gz")) == 2 and len(core.hits) == 2


# ---------------------------------------------------------------- meta
def test_meta_records_full_config(pool: Path, tmp_path: Path):
    """meta.json に生成 config が全部残る(層割当・model・温度・重複率・時刻)。"""
    _, core_url = _start_stub("stub-core")
    _, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == bb.SCHEMA_VERSION
    assert meta["prompt_version"] == bb.PROMPT_VERSION
    assert meta["temperature"] == 0.9 and meta["max_tokens"] == 200
    assert meta["seed"] == 42 and meta["api_mode"] == "chat"
    assert meta["groups"]["core"] == {
        "model": "stub-core", "served_name": "stub-core",
        "servers": [core_url], "layers": ["L1"], "used": True}
    assert meta["groups"]["mass"]["layers"] == ["L2"]
    assert meta["started_at"] and meta["finished_at"]
    by_layer = {x["layer"]: x for x in meta["layers"]}
    assert by_layer["L1"]["written"] == 6 and by_layer["L2"]["group"] == "mass"
    assert by_layer["L1"]["duplicate_rate"] == 0.0
    assert 40 <= by_layer["L1"]["length_chars"]["p50"] <= 200
    assert by_layer["L1"]["completion_tokens"] == 6 * 90


def test_duplicate_rate_is_measured(pool: Path, tmp_path: Path):
    """同一文を返し続けたら重複率がそのまま meta に出る(テンプレ化の検出口)。"""
    _, core_url = _start_stub("stub-core", lambda req, i: _good_text(0))
    _, mass_url = _start_stub("stub-mass", lambda req, i: _good_text(0))
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url, "--layers", "L1")
    stat = json.loads((out / "meta.json").read_text(encoding="utf-8"))["layers"][0]
    assert stat["unique_backstories"] == 1 and stat["duplicate_lines"] == 5
    assert stat["duplicate_rate"] > 0.8


def test_model_probe_recorded(pool: Path, tmp_path: Path):
    """--no-probe を外すと /v1/models の申告(id・max_model_len)が来歴に残る。"""
    _, core_url = _start_stub("stub-core")
    _, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    argv = ["--pool", str(pool), "--out", str(out), "--core-servers", core_url,
            "--core-model", "stub-core", "--mass-servers", mass_url,
            "--mass-model", "stub-mass", "--core-layers", "L1",
            "--concurrency", "2", "--progress-s", "0", "--limit", "1"]
    assert bb.main(argv) == 0
    probe = json.loads((out / "meta.json").read_text(encoding="utf-8")
                       )["groups"]["core"]["probe"]
    assert probe[0]["id"] == "stub-core" and probe[0]["max_model_len"] == 8192


# ---------------------------------------------------------------- 救済(--rescue)
def test_rescue_varies_seed_and_elements_per_attempt():
    """試行ごとに seed 列と要素組合せが変わる。★attempt=0 は従来と 1 バイト同一。"""
    rec = _rec("L1_00000000", "L1")
    base = bb.build_prompt(rec, 42)
    assert bb.build_prompt(rec, 42, attempt=0) == base       # 本番パスは不変
    assert bb.pick_elements("L1_00000000", 42, 0) == bb.pick_elements("L1_00000000", 42)
    shas = {bb.prompt_sha(*base[:2])}
    seeds = {bb.stable_seed(42, "L1_00000000"),
             bb.stable_seed(42, "L1_00000000", "backstory", "retry")}
    combos = set()
    for k in range(1, 6):
        s, u, _ = bb.build_prompt(rec, 42, attempt=k)
        assert bb.build_prompt(rec, 42, attempt=k) == (s, u, _)   # 決定論
        shas.add(bb.prompt_sha(s, u))
        seeds.add(bb.stable_seed(42, "L1_00000000", "backstory", f"rescue{k}"))
        combos.add(bb.pick_elements("L1_00000000", 42, k))
        assert "字数厳守" in u                                # short 対策の駄目押し
    assert len(shas) == 6 and len(seeds) == 7                 # 全部相異なる列
    assert len(combos) >= 3                                   # 切り口が回っている


def test_rescue_recovers_and_prunes_failed(pool: Path, tmp_path: Path):
    """救済成功 → 本体へ追記 + failed から除去(全部拾えたらファイル自体を消す)。"""
    out = tmp_path / "out"
    # 1 回目: 全件 short で落とす
    _, core_url = _start_stub("stub-core", lambda req, i: "短い経歴。")
    _, mass_url = _start_stub("stub-mass", lambda req, i: "短い経歴。")
    _run(pool, out, core_url, mass_url, "--layers", "L1")
    assert _lines(out / "L1.jsonl.gz") == []
    assert len((out / "L1.failed.jsonl").read_text(encoding="utf-8").splitlines()) == 6

    # 2 回目: rescue2 の seed でだけ合格文を返す = 1 回目の試行は落ち 2 回目で拾う
    ok_seeds = {bb.stable_seed(42, f"L1_{i:08d}", "backstory", "rescue2")
                for i in range(6)}
    core2, core2_url = _start_stub(
        "stub-core", lambda req, i: _good_text(i) if req["seed"] in ok_seeds else "短い。")
    rc = bb.main(["--pool", str(pool), "--out", str(out), "--layers", "L1",
                  "--core-layers", "L1", "--core-servers", core2_url,
                  "--core-model", "stub-core", "--rescue", "3",
                  "--concurrency", "4", "--progress-s", "0",
                  "--http-retries", "0", "--no-probe"])
    assert rc == 0
    lines = _lines(out / "L1.jsonl.gz")
    assert len(lines) == 6 and len(core2.hits) == 12          # 6 人 × 2 試行で決着
    for line in lines:
        assert line["rescue"] == 2
        assert line["seed"] == bb.stable_seed(42, line["pid"], "backstory", "rescue2")
        assert bb.check(line["backstory"]) is None
    assert not (out / "L1.failed.jsonl").exists()             # 全部回収 = 残さない
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "rescue"
    assert meta["rescue"] == {"attempts": 3, "targets": 6, "attempted": 6,
                              "rescued": 6, "remaining": 0}
    assert meta["layers"][0]["mode"] == "rescue"


def test_rescue_keeps_hopeless_pids_in_failed(pool: Path, tmp_path: Path):
    """全 attempt 落ちた個体だけが failed に残る(理由と試行回数を更新)。"""
    out = tmp_path / "out"
    _, core_url = _start_stub("stub-core", lambda req, i: "")
    _, mass_url = _start_stub("stub-mass", lambda req, i: "")
    _run(pool, out, core_url, mass_url, "--layers", "L1")
    core2, core2_url = _start_stub("stub-core", lambda req, i: "")
    rc = bb.main(["--pool", str(pool), "--out", str(out), "--layers", "L1",
                  "--core-layers", "L1", "--core-servers", core2_url,
                  "--core-model", "stub-core", "--rescue", "3",
                  "--concurrency", "4", "--progress-s", "0",
                  "--http-retries", "0", "--no-probe"])
    assert rc == 0
    assert _lines(out / "L1.jsonl.gz") == []
    assert len(core2.hits) == 18                             # 6 人 × 3 試行
    rows = [json.loads(x) for x in
            (out / "L1.failed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert {x["rescue_attempts"] for x in rows} == {3}
    assert {x["reason"] for x in rows} == {"empty"}
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["rescue"]["rescued"] == 0 and meta["rescue"]["remaining"] == 6


def test_rescue_is_idempotent_for_already_recovered(pool: Path, tmp_path: Path):
    """本体に既に在る pid は撃たずに failed から外すだけ(2 度目の救済は無害)。"""
    out = tmp_path / "out"
    _, core_url = _start_stub("stub-core", lambda req, i: "短い経歴。")
    _, mass_url = _start_stub("stub-mass", lambda req, i: "短い経歴。")
    _run(pool, out, core_url, mass_url, "--layers", "L1")
    # 本体へ 1 人だけ手で入れてから救済 → その 1 人は撃たれない
    body = _lines(out / "L1.jsonl.gz")
    assert body == []
    import gzip as _gz
    with _gz.open(out / "L1.jsonl.gz", "at", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": "L1_00000000", "backstory": _good_text(1),
                             "model": "stub-core", "seed": 1,
                             "prompt_sha": "x" * 16}, ensure_ascii=False) + "\n")
    core2, core2_url = _start_stub("stub-core", lambda req, i: "短い。")
    bb.main(["--pool", str(pool), "--out", str(out), "--layers", "L1",
             "--core-layers", "L1", "--core-servers", core2_url,
             "--core-model", "stub-core", "--rescue", "1", "--concurrency", "4",
             "--progress-s", "0", "--http-retries", "0", "--no-probe"])
    assert len(core2.hits) == 5                              # 6 - 既に在る 1 人
    rows = [json.loads(x) for x in
            (out / "L1.failed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {x["pid"] for x in rows} == {f"L1_{i:08d}" for i in range(1, 6)}
    stat = json.loads((out / "meta.json").read_text(encoding="utf-8"))["layers"][0]
    assert stat["already_in_body"] == 1 and stat["attempted"] == 5


# ---------------------------------------------------------------- 2 段運用
def test_mass_only_stage_needs_no_core_config(pool: Path, tmp_path: Path):
    """① mass 段: --layers で mass の層だけなら core の servers/model は未指定で良い。

    プローブもその段で使う群だけ引く(起動していないサーバを触りに行かない)。
    """
    mass, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    rc = bb.main(["--pool", str(pool), "--out", str(out), "--layers", "L2",
                  "--core-layers", "L1", "--mass-servers", mass_url,
                  "--mass-model", "stub-mass", "--concurrency", "4",
                  "--progress-s", "0", "--http-retries", "0"])
    assert rc == 0
    assert len(_lines(out / "L2.jsonl.gz")) == 8 and len(mass.hits) == 8
    assert not (out / "L1.jsonl.gz").exists()          # 包含フィルタ = L1 は走らない
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["groups"]["mass"]["used"] is True
    assert meta["groups"]["core"]["used"] is False
    assert "probe" in meta["groups"]["mass"]           # 使う群だけプローブ
    assert "probe" not in meta["groups"]["core"]
    assert [x["layer"] for x in meta["layers"]] == ["L2"]


def test_core_only_stage_needs_no_mass_config(pool: Path, tmp_path: Path):
    """② core 段: 同じ --out へ書き足す(層別ファイルなので互いに触らない)。"""
    mass, mass_url = _start_stub("stub-mass")
    core, core_url = _start_stub("stub-core")
    out = tmp_path / "out"
    assert bb.main(["--pool", str(pool), "--out", str(out), "--layers", "L2",
                    "--core-layers", "L1", "--mass-servers", mass_url,
                    "--mass-model", "stub-mass", "--concurrency", "4",
                    "--progress-s", "0", "--no-probe"]) == 0
    l2_bytes = (out / "L2.jsonl.gz").read_bytes()
    assert bb.main(["--pool", str(pool), "--out", str(out), "--layers", "L1",
                    "--core-layers", "L1", "--core-servers", core_url,
                    "--core-model", "stub-core", "--concurrency", "4",
                    "--progress-s", "0", "--no-probe"]) == 0
    assert len(_lines(out / "L1.jsonl.gz")) == 6 and len(core.hits) == 6
    assert (out / "L2.jsonl.gz").read_bytes() == l2_bytes   # 前段の成果物は不触
    assert len(mass.hits) == 8                              # 2 段目で再送しない
    # meta.json は上書きされるが、前段の来歴は previous_stages に畳んで残る
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert [x["layer"] for x in meta["layers"]] == ["L1"]
    assert len(meta["previous_stages"]) == 1
    prev = meta["previous_stages"][0]
    assert [x["layer"] for x in prev["layers"]] == ["L2"]
    assert prev["groups"]["mass"]["model"] == "stub-mass"


def test_unconfigured_group_with_routed_layer_is_a_clear_error(pool: Path,
                                                               tmp_path: Path):
    """設定の無いグループへ層がルーティングされたら、黙って走らせずに止める。"""
    _, mass_url = _start_stub("stub-mass")
    with pytest.raises(SystemExit) as err:
        bb.main(["--pool", str(pool), "--out", str(tmp_path / "out"),
                 "--layers", "L1,L2", "--core-layers", "L1",
                 "--mass-servers", mass_url, "--mass-model", "stub-mass",
                 "--progress-s", "0", "--no-probe"])
    msg = str(err.value)
    assert "core" in msg and "--core-model" in msg and "L1" in msg
    assert not (tmp_path / "out" / "L2.jsonl.gz").exists()   # 1 件も生成していない


# ---------------------------------------------------------------- レーン間契約
def test_output_is_readable_by_engine_side_store(pool: Path, tmp_path: Path):
    """生成物が**そのまま**エンジン側リーダで引ける(レーン間の契約ピン)。

    契約は 1 行の {"pid","backstory"} と <root>/<layer>.jsonl.gz の置き場だけ。
    エンジン側 module がまだ無い環境では skip(生成レーン単体では検収できるまま)。
    """
    store_mod = pytest.importorskip("society.world.backstory")
    _, core_url = _start_stub("stub-core")
    _, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    _run(pool, out, core_url, mass_url)
    assert (store_mod.PID_KEY, store_mod.TEXT_KEY) == ("pid", "backstory")
    store = store_mod.BackstoryStore(out)
    for line in _lines(out / "L1.jsonl.gz"):
        assert store.get(line["pid"], "L1") == line["backstory"]
    assert store.get("L1_99999999", "L1") == ""      # 欠損は空文字(例外にしない)


# ---------------------------------------------------------------- 実プール
@pytest.mark.skipif(not (REPO_ROOT / "data" / "persona_pool_v2" / "L1").is_dir(),
                    reason="data/persona_pool_v2 が無い環境")
def test_dryrun_on_real_pool_100(tmp_path: Path):
    """実プールの part を読み、スタブで 100 人分を生成できる(形式確定の検収)。"""
    core, core_url = _start_stub("stub-core")
    _, mass_url = _start_stub("stub-mass")
    out = tmp_path / "out"
    argv = ["--pool", str(REPO_ROOT / "data" / "persona_pool_v2"), "--out", str(out),
            "--core-servers", core_url, "--core-model", "stub-core",
            "--mass-servers", mass_url, "--mass-model", "stub-mass",
            "--core-layers", "L1", "--layers", "L1", "--limit", "100",
            "--concurrency", "8", "--progress-s", "0", "--no-probe"]
    assert bb.main(argv) == 0
    lines = _lines(out / "L1.jsonl.gz")
    assert len(lines) == 100
    assert all(x["pid"].startswith("L1_") for x in lines)
    assert all(bb.check(x["backstory"]) is None for x in lines)
    assert len({x["prompt_sha"] for x in lines}) == 100      # 個体ごとに違うプロンプト
    # resume: 2 回目は 0 リクエスト
    n = len(core.hits)
    assert bb.main(argv) == 0
    assert len(core.hits) == n and len(_lines(out / "L1.jsonl.gz")) == 100
