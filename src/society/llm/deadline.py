"""LLM 1 呼の**絶対時限**(hard deadline)— 第86バッチ保守 M-1(2026-08-03)。

なぜ要るか(実測)
------------------
2026-08-02 夜間ランで **1 呼が 1 時間 47 分**張り付いた。`model.timeout_s`(既定 120s)は
`urllib.request.urlopen(..., timeout=...)` に渡る **ソケット単位のタイムアウト**であって、
「無通信が続いた時間」しか測れない。トークンが細々と流れ続ける病的生成(サーバが本文を
少しずつ書き続ける)では**無通信区間が一度も 120s に達しない**ので永久に発火しない。
10 日ランでは 1 呼の張り付きがそのままラン全体の停止を意味する(watchdog の stall 検知が
働くまで GPU が空回りする)ので、**呼び出し開始からの経過時間**で切る層をここに置く。

方式(なぜタイマー方式か)
--------------------------
- backend は 4 本とも標準ライブラリ(`urllib`)で `stream=false` の 1 リクエストを投げ、
  `resp.read()` で本文を一括読みする。したがって「読取ループの中で経過を見る」口が無い
  (read() は 1 回の呼び出しでブロックする)。requests のストリーミング読取も使っていない。
- そこで **別スレッドのタイマーでソケットを shutdown する**。ブロック中の recv は
  shutdown で必ず解ける(戻り値 0 か例外)。ワーカースレッドに読取を退避させる方式は、
  時限超過後もスレッドが 1 時間 47 分ぶん生き残る(= 10 日ランでスレッドが溜まる)ので採らない。
- 正常呼(数秒)ではタイマーは**必ず cancel される**= 発火しない = 挙動は 1 バイトも変わらない。
  過去の実 LLM 呼分布は 3.05 s/呼 なので既定 300s は 100 倍のマージンがある。

観測(deadline_exceeded カウンタ)
----------------------------------
超過はプロセス内のカウンタに積む(`exceeded_count()`)。Simulation.finalize が
`summary.json` の `llm_deadline_exceeded` に載せ(**0 のときはキー自体を出さない**=
既存ランの summary とバイト一致)、`scripts/watchdog_llm.py` がそれを読んで警告する。
CachedLLM/Router を貫通させずにここへ集約するのは、超過が「どの層でも起こりうるが
必ずこの 1 関数を通る」性質のものだからである(呼数・キャッシュ統計とは源が違う)。

決定論への影響
--------------
mock backend はここを通らない(HTTP を使わない)。実 LLM は元々 wall-clock 依存なので
時限の有無で再現性の等級は変わらない(再現性の実体は llm_cache = D13)。
"""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

DEFAULT_DEADLINE_S = 300.0      # conf: model.call_deadline_s(0 以下 = 無効 = 従来どおり)

_lock = threading.Lock()
_exceeded = 0


class DeadlineExceeded(TimeoutError):
    """1 呼の絶対時限を超えた。

    `TimeoutError` を継承するのが要点: backend 側の except 節は既に
    `(urllib.error.URLError, TimeoutError, OSError, ...)` を拾って
    "__*_error__: ..." を返す = **既存のタイムアウト→fallback 経路にそのまま合流する**
    (新しい失敗の型を上位へ増やさない)。
    """

    def __init__(self, elapsed_s: float, deadline_s: float):
        self.elapsed_s = float(elapsed_s)
        self.deadline_s = float(deadline_s)
        super().__init__(
            f"call deadline {self.deadline_s:.0f}s exceeded "
            f"(elapsed {self.elapsed_s:.1f}s; model.call_deadline_s)")


def exceeded_count() -> int:
    """このプロセスで絶対時限に掛かった呼の総数。"""
    with _lock:
        return _exceeded


def reset_counter() -> None:
    """テスト用。ラン本体からは呼ばない。"""
    global _exceeded
    with _lock:
        _exceeded = 0


def _note() -> None:
    global _exceeded
    with _lock:
        _exceeded += 1


def _close(resp) -> None:
    """best-effort な close(テストのスタブ応答は close を持たないことがある)。"""
    fn = getattr(resp, "close", None)
    if fn is None:
        return
    try:
        fn()
    except Exception:                     # noqa: BLE001 (後始末で例外を上げない)
        pass


def _abort(resp) -> None:
    """別スレッドから、ブロック中の read を確実に解く(shutdown → close)。

    close だけでは CPython の socket が参照カウントで生きているとブロックが解けないため、
    先に shutdown で FIN/RST を送る。どちらも best-effort(既に閉じていれば OSError)。
    """
    sock = None
    try:
        sock = resp.fp.raw._sock          # http.client.HTTPResponse の実体
    except Exception:                     # noqa: BLE001 (実体が違えば socket は諦めて close のみ)
        pass
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    _close(resp)


def request_json(req: urllib.request.Request, *, timeout_s: float,
                 deadline_s: float | None = DEFAULT_DEADLINE_S) -> dict:
    """`urlopen` + 本文読み切りを「呼び出し開始からの絶対時限」で囲って JSON を返す。

    - `timeout_s`  … 従来どおりのソケット単位タイムアウト(無通信区間の上限)。
    - `deadline_s` … 呼び出し開始からの絶対時限。None / 0 以下で**完全に無効**
                     (= 従来と同一のコード経路)。
    - `urllib.error.HTTPError` は**素通し**する(呼び手が 404/400 で分岐するため)。
    """
    dl = float(deadline_s or 0.0)
    if dl <= 0.0:                                   # 無効 = 従来どおりの 1 行
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    start = time.monotonic()
    # 接続+ヘッダも時限の内側(残り時間とソケット timeout の小さい方で待つ)
    try:
        resp = urllib.request.urlopen(
            req, timeout=max(0.001, min(float(timeout_s), dl)))
    except urllib.error.HTTPError:
        raise                                       # 呼び手の code 分岐へ素通し
    except Exception:
        if time.monotonic() - start >= dl:
            _note()
            raise DeadlineExceeded(time.monotonic() - start, dl) from None
        raise

    aborted = threading.Event()

    def _fire() -> None:
        aborted.set()
        _abort(resp)

    timer = threading.Timer(max(0.0, dl - (time.monotonic() - start)), _fire)
    timer.daemon = True
    timer.start()
    try:
        raw = resp.read()
    except Exception:
        # 時限で叩き切ったときの例外は種類が環境依存(OSError / http.client.IncompleteRead /
        # ConnectionAbortedError …)。aborted が立っているときだけ超過として翻訳し、
        # それ以外は**元の例外のまま**上げる(通常のネットワーク障害を時限に化けさせない)。
        if aborted.is_set():
            _note()
            raise DeadlineExceeded(time.monotonic() - start, dl) from None
        raise
    finally:
        timer.cancel()
        _close(resp)
    if aborted.is_set():          # 読み切った瞬間にタイマーが発火した競合(経過は既に超過)
        _note()
        raise DeadlineExceeded(time.monotonic() - start, dl)
    return json.loads(raw.decode("utf-8"))
