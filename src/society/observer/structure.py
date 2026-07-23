"""社会構造の内生変動レンズ(第56バッチ タスクB 2026-07-24。既定 OFF)。

「強制トリガーなしで社会構造は変化するのか、固着するのか」を正面から測る観測レンズ。
外部アドバイスの「関係性の固定化」が実際に起きるかを、**介入せず**観測する。**観測のみ**=
構造を揺らす/固める介入は一切しない(危機トリガー不採用の日常観察方針=Entry 47)。

  設計正典: docs/closed-world-daily-observation.md §3 タスクB / devlog Entry 47(Fable 精査で確定した
  設計条件)。lens.py(第50バッチ)・deviation.py(第55バッチ)の作法(build_cfg 正準化・cfg_of 遅延
  キャッシュ・observe 当日タリー・DEFAULT を単一の源)をそのまま踏襲する。

--- Fable 精査で確定した設計条件(変更不可)-------------------------------------------------
(1) **in-sim L2 は当日イベントだけで計算できる軽量スカラーのみ**にする。edge 組み替え率(日次の
    新規形成/断絶/風化の件数・率)を relation_tier / relation_break イベントから当日カウントする。
    ★**前日状態をシム側に持たない**(checkpoint 肥大と resume 面積を増やさない)。当日タリーは lens/
      deviation と同型の watermark で新規イベントだけ走査。母数(活性関係数)は当日境界で 1 回だけ
      live 状態(agent.mem.relations の tier≥1)を数えてキャッシュ=毎 step の全走査を避ける。
(2) 4 スカラー列(relations.py のイベント種と台帳へ忠実に対応):
      ・edges_formed  = 当日の relation_tier{tier==1}(新規に tier≥1 到達=紐帯の形成。1→2/2→3 の
                        深化は「組み替え」でないので数えない)。
      ・edges_broken  = 当日の relation_break{cause=="contact"}(能動的な負交流による断絶・悪化=tier降格)。
      ・edges_decayed = 当日の relation_break{cause=="absence"}(長期不在による風化=decay の tier降格)。
      ・edge_churn_rate = (formed + broken + decayed) / max(活性関係数, 1)。母数=当日境界の tier≥1 関係数。
(3) **順位相関(Kendall τ)・中心性入れ替わり・コミュニティ変化・固着検知は事後分析層**
    (scripts/analyze_structure.py)が L1+L3 から計算する。ここ(in-sim)は軽量スカラーに徹する
    (前日状態=前日ランキング/前日グラフをシム側に持たない=resume バイト一致を守る)。
    ★コミュニティ検出は Louvain(networkx)でなく既存の決定論 LPA(observer/measure.communities)を
      事後層が再利用する(新規依存なし)。ここには検出ロジックを置かない。

--- R1 ドクトリン ---------------------------------------------------------------------------
既定 OFF(lens.structure.enabled=false): observe() が即 None・L2 に列を足さない=L1/L2/乱数・
  ゴールデンがバイト一致で不変。ON でも「読むだけ」= 行動・乱数・LLM 呼を一切増やさない。ロジックは
  observer 層(本 module)+ conf(lens.structure ブロック)に閉じ、engine/cognition/world/factors/
  actions/labeling には構造語を一切書かない(no-fingerprint)。決定論・RNG stream 不要。
"""
from __future__ import annotations

# relation_break の cause 分類(relations.py: note_contact→"contact" / decay_day→"absence")。
# 能動的な負交流の断絶と、長期不在による風化(decay)を分けて数える。
_BREAK_ABSENCE = "absence"   # 風化(decay_day)= edges_decayed
# cause が absence 以外(既定 "contact")の relation_break は edges_broken に数える。


def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の lens.structure ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    enabled= 独立マスター(value4/motives/trust/deviation とは別の observable。lens.enabled に依存しない=
    構造レンズだけ ON にできる)。将来のパラメータ拡張点として dict を返す(現状 enabled のみ)。"""
    raw = _to_plain(raw) or {}
    raw = dict(raw)
    return {"enabled": bool(raw.get("enabled", False))}


def cfg_of(sim) -> dict:
    """structure 設定を返す(初回のみ sim.cfg.lens.structure から遅延構築してキャッシュ)。

    simulation.py は読み取り専用のため cfg は本 module が遅延構築する(lens/deviation.cfg_of と同型)。
    キャッシュ属性 sim.structcfg は L1/L2/L3/乱数に一切現れない=既定 OFF のバイト一致を壊さない。"""
    c = getattr(sim, "structcfg", None)
    if c is None:
        raw = None
        try:
            raw = (sim.cfg.get("lens", None) or {}).get("structure", None)
        except Exception:
            raw = None
        c = build_cfg(raw)
        sim.structcfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 母数(活性関係数): 当日境界で 1 回だけ live 状態を数える(前日状態を持たない=毎 step 走査しない)
# --------------------------------------------------------------------------- #
def _active_ties(sim) -> int:
    """現時点で tier≥1 の関係(有向台帳エントリ)の総数。relations OFF(tier キー無し)なら 0。

    有向カウント(a→b と b→a を別に数える)。churn イベント(actor 視点で記録)も有向なので比が整合する。"""
    n = 0
    for a in getattr(sim, "agents", []):
        mem = getattr(a, "mem", None)
        if mem is None:
            continue
        for rel in getattr(mem, "relations", {}).values():
            if int(rel.get("tier", 0)) >= 1:
                n += 1
    return n


def _fresh_state() -> dict:
    return {"day": -1, "formed": 0, "broken": 0, "decayed": 0, "active": 0}


# --------------------------------------------------------------------------- #
# observe(L2 用): 当日(暦日)の形成/断絶/風化を積み、母数をキャッシュして返す(OFF は None)
# --------------------------------------------------------------------------- #
def observe(sim) -> dict | None:
    """当日の edge 組み替え(formed/broken/decayed)を集計して state を返す(OFF は None=列なし=L2 不変)。

    lens/deviation.observe と同型: collect(sim) が step 末に 1 回だけ呼ぶ経路に載り、前回処理済み総数
    sim._struct_processed 以降の新規イベントだけを走査(idempotent・暦日境界でタリー+母数をリセット)。
    決定論・乱数/LLM 呼ゼロ・読むだけ。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)   # ランを通じ単調増加
    processed = int(getattr(sim, "_struct_processed", 0))
    new = total - processed
    if new < 0:
        new = 0
    if new > len(events):          # flush 前に処理済みのはず。安全側=buffer 内だけ数える
        new = len(events)
    st = getattr(sim, "_struct_state", None)
    if st is None:
        st = _fresh_state()
        sim._struct_state = st
    for e in events[len(events) - new:]:
        d = int(e.sim_min) // 1440
        if d != st["day"]:         # 暦日が進んだ → 当日タリーをリセット + 母数を live から数え直す
            st["day"] = d
            st["formed"] = 0
            st["broken"] = 0
            st["decayed"] = 0
            st["active"] = _active_ties(sim)   # 当日 1 回のみ(毎 step の全走査を避ける)
        k = e.kind
        if k == "relation_tier":
            if int(e.payload.get("tier", 0)) == 1:      # 0→1=新規紐帯(深化 1→2/2→3 は数えない)
                st["formed"] += 1
        elif k == "relation_break":
            if e.payload.get("cause") == _BREAK_ABSENCE:
                st["decayed"] += 1                       # 長期不在の風化(decay)
            else:
                st["broken"] += 1                        # 能動的な負交流の断絶・悪化
    sim._struct_processed = total
    return st


def scalars(sim) -> dict | None:
    """observe の state → L2 全体スカラー 4 列(OFF は None)。前日状態不要=当日イベント+当日母数のみ。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    st = observe(sim)
    if st is None:
        return None
    formed = int(st["formed"])
    broken = int(st["broken"])
    decayed = int(st["decayed"])
    churn = formed + broken + decayed
    active = int(st["active"])
    return {
        "edges_formed": formed,                          # 当日の新規 tier≥1 到達(紐帯の形成)
        "edges_broken": broken,                          # 当日の relation_break(能動的な断絶・悪化)
        "edges_decayed": decayed,                        # 当日の風化(長期不在の decay=tier降格)
        "edge_churn_rate": round(churn / max(active, 1), 6),  # 組み替え率(母数=活性関係数)
    }
