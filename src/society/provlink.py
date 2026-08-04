"""行為 → 思考の来歴リンク IF-1(observer.llm_link・**既定 OFF**)。

正典
----
- docs/research/llm-world-interface-audit.md §4(観測者IF の欠落)・§5-3・§6 IF-1
- docs/research/if-lane-research.md §4(W3C PROV-DM の ``wasInformedBy`` +
  ``prov:role``。§4-3 の設計示唆 1〜7)
- docs/plans/if-sv-p4-plan.md 「IF-A」行

何を解く問題か
--------------
本シムは L1(行為 189 種)と l1b_llm(思考 = LLM 呼)を別々に持っているが、
**行為イベント側に llm_call_id が無い**(思考系イベントにしか付いていない)。
そのため「この発話を決めたのはどの思考か」は ``(step, agent_id)`` の join に
頼るしかなく、同 step に複数の呼が発火すると原理的に曖昧になる。

PROV-DM の語彙で言えば、これは **Activity(行為)← Activity(思考)の
``wasInformedBy`` 辺が 1 本欠けている**状態にあたる(§4-2 の対応表)。
本 module はその 1 本を通す。残りの 2 本(``state_update.cause`` =
状態差分 ← 行為 / ``Item.transmissions`` = entity ← entity)は既にあるので、
この辺が通った時点で「世界の変化 → 原因の行為 → その思考」が事後に辿れる。

設計(§4-3 の示唆にそのまま対応)
--------------------------------
1. **単独値ではなく ``(llm_call_id, role)`` の対**にする。同 step 複数発火の
   曖昧さは PROV が「関係に識別子と ``prov:role`` を付けられる」ことで解いて
   いる問題そのもの。
2. ``role`` の語彙は **``l1b_llm.parquet`` の ``purpose`` と同一**にし、
   **新語彙を 1 つも作らない**(§4-3-1)。実際に入る値は
   ``social / reply / dm / post / solo / novel_place / congestion /
   unknown_word``(熟慮経路 = ``_llm_speak`` の trigger)と ``plan``
   (朝の計画経路 = day_plan のブロック実行)。
   → **役割そのものが経路種別を表す**: role="plan" の行為は「朝立てた計画の
     ブロックを実行した」もので、role=それ以外は「その場の熟慮が決めた」もの。
3. **捕集(capturer)と組み立て(builder)を分離する**(SIMPROV。§4-3-6)。
   シム側は payload に識別子を載せるだけで、**DAG は持たない**。グラフ構成は
   事後の解析スクリプトの仕事。これが「観測がシムを変えない」の実装形でもある。
4. **既定 OFF + OFF ではキー自体を生やさない**(第75 dormant / 第86 day_plan と
   同じ流儀)。OFF では ``_prov`` 一時キーが 1 度も積まれないので、
   ``_apply`` の来歴スコープに入る経路が**構造的に**存在しない
   = ゴールデン L1 バイト一致。
5. **replay 契約は変わらない**(Fowler の event sourcing 警告。§4-3-5)。
   本 module はログの**書き込み側**にしか触れず、外部呼び出しを 1 本も足さない。
   ``llm_cache`` の「ミス即例外」がこれまでどおり replay の門番であり続ける。

正直な限界
----------
- **刻むのは「その行為者自身が出したイベント」だけ**(``_apply`` 中に
  ``event.agent_id == 行為者`` のものだけ)。聞き手側の ``hear`` /
  ``opinion_shift`` などは **別の主体の activity** なので刻まない
  (PROV では ``wasAssociatedWith`` の agent が違う)。話者→聞き手の辺は
  既存の ``speak.payload.hearers`` と ``transmission`` が担っている。
- ``agent_id=-1`` の世界イベント(``delivery_trip`` / ``stock_low`` など)は
  行為の副次で出ても刻まれない(上の規則の帰結)。
- **方針キャッシュ(cognition.policy_cache)で再利用された行為には辺が付かない**。
  その step には LLM 呼が存在しないので指す先が無い(捏造しない)。既定 OFF。
- 整数 ID 化(§4-3-3「25万体で文字列 UUID はサイズが爆発する」)は**採らなかった**。
  ``llm_call_id`` は既に L1 の固定列として存在し(dtype=string)、l1b 側も
  同じ文字列なので、ここで別の整数体系を作ると join 相手が 2 つになる。
  サイズ実測に基づく型変更は L1 スキーマ全体の話なので別バッチの仕事として残す。
"""
from __future__ import annotations

# 行為 dict へ一時的に積む鍵。**_apply の入口で必ず pop される**ので、
# 世界状態にも L1 payload にも残らない(残っていたらバグ)。
PROV_KEY = "_prov"

# payload へ出す role の鍵(Event のスキーマは変えない = §4-3-4 の判断)。
ROLE_KEY = "llm_role"


def enabled(sim) -> bool:
    """observer.llm_link が実効か(初回のみ読んでキャッシュ)。

    キャッシュ属性 ``_llm_link_on`` は L1/L2/乱数・checkpoint のいずれにも出ない。
    """
    on = getattr(sim, "_llm_link_on", None)
    if on is None:
        try:
            on = bool((sim.cfg.get("observer", None) or {}).get("llm_link", False))
        except Exception:                          # noqa: BLE001 (旧 config 互換)
            on = False
        sim._llm_link_on = on
    return on


def stamp(sim, action, call_id, role: str) -> None:
    """行為 dict へ ``(llm_call_id, role)`` を積む(**OFF ではキー自体を作らない**)。

    ルール層(routine / engaged のテンプレ応答 / 方針キャッシュ再利用)由来の行為は
    そもそもここを通らないので None のまま = 辺が付かない(捏造しない)。
    """
    if not isinstance(action, dict) or not call_id or not enabled(sim):
        return
    action[PROV_KEY] = (str(call_id), str(role))


def take(action) -> tuple[str, str] | None:
    """一時キーを取り出して**消す**。OFF では常に None(キーが無い)。"""
    if not isinstance(action, dict):
        return None
    prov = action.pop(PROV_KEY, None)
    if not prov:
        return None
    return str(prov[0]), str(prov[1])
