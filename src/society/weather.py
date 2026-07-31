"""天気(第7バッチ 2026-07-07。ユーザー要望 / 第80バッチ W2 で生成型実データ化)。

その日の天気を日次で決定論生成し、エージェント行動・気分に影響させる(グローバル
文脈・全員共通・k 非依存)。天気は新 stream ``sim.hub.stream("weather", day_index)``
から**1日1回だけ**引く(既存 stream の draw 順に影響しない=決定論・ゴールデン不変)。

天気→不快感(grievance)は factors 係数レイヤー経由(R9)。このモジュールは grievance の
係数(rain_grievance)を保持し、雨天など不快な天気の日に不透明な magnitude を返すだけ
(scheduler がそれを factors.update.on_weather に渡す)。乱数は増やさない(天気は既に確定
=決定論的加算)。

3つのモード(第80バッチ W2。正典 docs/research/weather-generator-design.md §4)
-----------------------------------------------------------------------------
``weather.mode``:

  ``synthetic``(既定) 現行のまま(月別テーブル + ±3℃ 一様)。**1バイトも挙動を変えない**。
                      ★ただし構造的欠陥がある: 8月は 32±3 なので**上限 35℃・36℃以上は
                        確率0**、日間自己相関ゼロ、年ごとの差もほぼ無い(設計書 §0-2)。
  ``generated``       較正済み生成器(WGEN 系 3状態マルコフ + Matalas 2変量 AR(1) + 年効果)。
                      パラメータは静的ファイル。**ラン開始時に全日数を一括生成してメモ化**し、
                      AR(1) の状態を sim に持たせない = ``checkpoint.py`` は無改修で
                      resume しても同一系列になる(再構築が (master_seed, params) の純関数)。
                      新 stream ``"weather_gen"`` を使うので synthetic の draw 順は不変。
  ``table``           凍結した実測日別データを **実日付**(暦の start_date 基準)で引く。
                      乱数を1つも引かない=完全決定論。反実仮想はできない(1つの実現のみ)。

較正済みでない月・凍結範囲外の日付・暦 OFF では **エラーにせず ``synthetic`` へ落とす**
(黙って落とさない: 警告ログ + summary/run_manifest の来歴ブロックに件数と理由を残す)。

★ このファイルは src/society 直下(engine/cognition/world 検査対象外)なので、因子語を
  含む係数ロジックはここに置いてよい(no-fingerprint 契約に触れない)。
★ 場所の知識(パラメータファイルのパス)は持たない: envpack.climate.gen_params /
  envpack.climate.table から降ってくる(基盤に東京固有のパスを残さない)。
既定 OFF(enabled=false)= 一切引かず・イベントも出さず・プロンプトにも載らない(不変)。
"""
from __future__ import annotations

import logging

from .world import calendar as _calendar

log = logging.getLogger("society.weather")

# 月(1-12)→ (最高気温中心, 最低気温中心, 雨の重み, 雪の重み)。
# 梅雨(6-7)は雨増、盛夏(7-8)は高温、冬(12-2)は寒く低確率で雪。現実の東京の月別気候の近似。
_MONTH_CLIMATE = {
    1: (10, 2, 0.15, 0.08), 2: (11, 2, 0.15, 0.06), 3: (15, 6, 0.22, 0.0),
    4: (20, 10, 0.25, 0.0), 5: (24, 14, 0.25, 0.0), 6: (26, 18, 0.45, 0.0),
    7: (30, 23, 0.42, 0.0), 8: (32, 25, 0.30, 0.0), 9: (27, 20, 0.35, 0.0),
    10: (22, 14, 0.25, 0.0), 11: (17, 9, 0.20, 0.0), 12: (12, 4, 0.15, 0.05),
}
# 季節なし(calendar OFF)の既定気候。ほぼ一様寄りの穏やかな分布。
_NEUTRAL_CLIMATE = (21, 12, 0.22, 0.0)

# 悪天候(不快感の源)とみなす天気。
_BAD_CONDS = {"雨", "雪"}

# 天候の決め方(mode)。既定 synthetic = 現行と完全同一経路。
MODES = ("synthetic", "generated", "table")
# 猛暑日(気象庁の定義: 日最高気温 35℃以上)。extra_prompt_fields の1行に使う。
_HOT_DAY_C = 35


def build_cfg(raw: dict | None, monthly: dict | None = None,
              neutral: list | tuple | None = None,
              gen_params: str | None = None,
              table_file: str | None = None) -> dict:
    """conf の weather ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    月別気候テーブル(monthly)と季節なし既定(neutral)、および生成器パラメータ /
    実測表のパス(gen_params / table_file)は envpack(場所の値)から受ける。
    未指定なら generic フォールバック(_MONTH_CLIMATE / _NEUTRAL_CLIMATE)を使う。

    ``mode`` が synthetic 以外かつ ``enabled`` のときだけ静的ファイルを読み込み、
    改竄検知(payload_sha256 の照合)と来歴(sha256)の採取をここで1回だけ行う。
    """
    raw = dict(raw or {})
    mon = {int(k): tuple(v) for k, v in (monthly or {}).items()} or _MONTH_CLIMATE
    neu = tuple(neutral) if neutral else _NEUTRAL_CLIMATE
    mode = str(raw.get("mode", "synthetic")).strip().lower()
    if mode not in MODES:
        raise ValueError(
            f"weather.mode='{mode}' は未知(有効値: {', '.join(MODES)})。既定 synthetic = 現行動作。")
    enabled = bool(raw.get("enabled", False))
    cfg = {
        "enabled": enabled,
        "mode": mode,
        # 雨天の日に grievance を上げる係数(factors 経由の加算量)。★ON 推奨: 0.01。
        # 既定 0.0 = 天気 ON でも grievance は動かさない(天気イベント・文脈行のみ)。
        "rain_grievance": float(raw.get("rain_grievance", 0.0)),
        # true で weather_line に最低気温・湿度・猛暑日・暑さ指数(推定)を足す。
        # ★プロンプトが変わる。synthetic では**無視する**(既定モードの文言は不変)。
        "extra_prompt_fields": bool(raw.get("extra_prompt_fields", False)),
        "monthly": mon,       # 月(int)→ (最高, 最低, 雨重み, 雪重み)。envpack.climate.monthly
        "neutral": neu,       # 季節なし既定。envpack.climate.neutral
        "gen_params_file": (str(gen_params) if gen_params else None),
        "table_file": (str(table_file) if table_file else None),
        "params": None,       # 較正パラメータ(generated 時のみ)
        "table": None,        # 実日付 → 実測レコード(table 時のみ)
        "table_md": None,     # 月日 → その月日の最新年のレコード(凍結範囲外の年の受け皿)
        "provenance": None,   # 入力データの来歴(summary / run_manifest へ)
    }
    if enabled and mode != "synthetic":
        _load_sources(cfg)
    return cfg


def _load_sources(cfg: dict) -> None:
    """generated / table が要る静的ファイルを読み、来歴(sha256)を採る。"""
    from . import weather_gen as _wg
    prov: dict = {"mode": cfg["mode"], "files": {}}
    if cfg["mode"] == "generated":
        path = cfg["gen_params_file"]
        if not path:
            raise ValueError(
                "weather.mode=generated だが較正パラメータのパスが無い"
                "(envpack.climate.gen_params を env / conf 側で指定すること)")
        doc = _wg.load_params(path)
        cfg["params"] = doc
        meta = doc.get("meta") or {}
        prov["weather_params_sha256"] = meta.get("payload_sha256")
        prov["weather_source_sha256"] = meta.get("source_payload_sha256")
        prov["fit_window"] = meta.get("fit_window")
        prov["calibrated_months"] = sorted(int(m) for m in (doc.get("months") or {}))
        prov["files"]["gen_params"] = {
            "path": path,
            "file_sha256": _wg.file_sha256(_wg.resolve_path(path)),
            "payload_sha256": meta.get("payload_sha256"),
            "attribution": meta.get("source_attribution"),
        }
    else:                                       # table
        path = cfg["table_file"]
        if not path:
            raise ValueError(
                "weather.mode=table だが実測データのパスが無い"
                "(envpack.climate.table を env / conf 側で指定すること)")
        doc = _wg.load_table(path)
        cfg["table"] = _wg.table_by_date(doc)
        cfg["table_md"] = _wg.table_latest_by_monthday(doc)
        meta = doc.get("meta") or {}
        prov["weather_source_sha256"] = meta.get("payload_sha256")
        prov["table_dates"] = [min(cfg["table"]), max(cfg["table"])]
        prov["files"]["table"] = {
            "path": path,
            "file_sha256": _wg.file_sha256(_wg.resolve_path(path)),
            "payload_sha256": meta.get("payload_sha256"),
            "attribution": ((meta.get("source") or {}).get("attribution")),
        }
    cfg["provenance"] = prov


def _sample(rng, month: int | None, monthly: dict, neutral) -> dict:
    """1日ぶんの天気を決定論生成(rng は当日専用 stream)。"""
    hi_c, lo_c, rain_w, snow_w = (
        monthly[month] if month is not None else neutral)
    r = float(rng.random())
    if r < snow_w:
        cond = "雪"
    elif r < snow_w + rain_w:
        cond = "雨"
    else:                                    # 残りを 曇(4割)/ 晴(6割)に割る
        span = 1.0 - snow_w - rain_w
        rem = r - (snow_w + rain_w)
        cond = "曇" if rem < span * 0.4 else "晴"
    temp_hi = int(hi_c + int(rng.integers(-3, 4)))     # ±3℃ の日毎ゆらぎ
    temp_lo = int(lo_c + int(rng.integers(-3, 4)))
    if temp_lo > temp_hi:                    # 最低が最高を超えない
        temp_lo = temp_hi
    return {"cond": cond, "temp_hi": temp_hi, "temp_lo": temp_lo}


# --------------------------------------------------------------------------- #
# generated / table(第80バッチ W2)
# --------------------------------------------------------------------------- #
def _stats(sim) -> dict:
    st = getattr(sim, "_weather_stats", None)
    if st is None:
        st = {"generated": 0, "table": 0, "table_year_remapped": 0,
              "synthetic_fallback": 0, "fallback_reasons": {}}
        sim._weather_stats = st
    return st


def _fallback(sim, reason: str) -> None:
    st = _stats(sim)
    st["synthetic_fallback"] += 1
    st["fallback_reasons"][reason] = st["fallback_reasons"].get(reason, 0) + 1
    if st["fallback_reasons"][reason] == 1:       # 同じ理由の警告は1回だけ(黙らせない/溢れさせない)
        log.warning("[weather] mode=%s だがこの日は合成へフォールバックした(理由: %s)。"
                    "件数は summary.json / run_manifest.json の weather ブロックに残る。",
                    sim.weathercfg.get("mode"), reason)


def _planned_days(sim) -> int:
    """このランが触りうる日数(+1 の余裕)。系列は逐次生成なので長めでも先頭は不変。"""
    try:
        n_steps = int(sim.cfg.run.n_steps)
    except Exception:                            # noqa: BLE001 (素の sim スタブ)
        n_steps = 144
    dt = int(getattr(getattr(sim, "clock", None), "step_minutes", 10) or 10)
    return max(1, -(-n_steps * dt // 1440)) + 1


def _generated_series(sim, wcfg: dict, anchor_dom: int, need_len: int) -> list[dict]:
    """全日数ぶんの系列を1本の stream から決定論生成してメモ化する(状態は持たない)。

    ★resume 安全: 再構築は (master_seed, params, anchor_dom) だけの関数で、生成は逐次
      なので **長さを伸ばしても先頭 n 日は不変**(prefix 安定)。よって途中から再開しても
      同じ系列になる = checkpoint に AR(1) の状態を載せる必要が無い。
    """
    from . import weather_gen as _wg
    ser = getattr(sim, "_weather_series", None)
    if ser is not None and len(ser) >= need_len:
        return ser
    month_key = str(_wg_anchor_month(sim))
    pm = wcfg["params"]["months"][month_key]
    prep = _wg.prepare(pm)
    n = max(need_len, _planned_days(sim))
    rng = sim.hub.stream("weather_gen", 0)       # ★新キー: 既存 "weather" の draw 順は不変
    seg = _wg.generate_segment(rng, prep, n, day_start=anchor_dom)
    ser = _wg.day_records(seg, prep)
    sim._weather_series = ser
    sim._weather_series_clipped = seg["n_clipped"]
    return ser


def _wg_anchor_month(sim) -> int:
    """系列を張る基準の月(= ラン初日の月)。1ランに AR(1) 連鎖は1本だけ張る。"""
    return _calendar.date_of(sim.calendarcfg, 0).month


def _nonsynthetic_for(sim, wcfg: dict, day_index: int) -> dict | None:
    """generated / table の当日レコード。使えない日は None(=呼び出し側が合成へ落とす)。"""
    cal = getattr(sim, "calendarcfg", None)
    if not (cal and cal.get("enabled")):
        # 暦が無いと「何月何日か」が決まらない=較正済みの月かどうかも判定できない。
        _fallback(sim, "calendar_off")
        return None
    date = _calendar.date_of(cal, day_index * 1440)
    if wcfg["mode"] == "table":
        from . import weather_gen as _wg
        iso = date.isoformat()
        rec = (wcfg["table"] or {}).get(iso)
        remapped = False
        if rec is None:
            # 暦の年が凍結範囲(例: 1996-2025)の外なら「同じ月日の**最新年**」へ回す。
            # ユーザーの DT 定義(ある時点の現実スナップショットが舞台・実時刻との同期は不要)
            # に沿った縮退。★年を差し替えた事実は必ず来歴に残す(黙って別年の値を使わない)。
            rec = (wcfg["table_md"] or {}).get(f"{date.month:02d}-{date.day:02d}")
            remapped = rec is not None
        if rec is None:
            _fallback(sim, "date_not_in_frozen_table")     # 凍結していない月(=8月以外)
            return None
        st = _stats(sim)
        st["table"] += 1
        if remapped:
            st["table_year_remapped"] = st.get("table_year_remapped", 0) + 1
            if st["table_year_remapped"] == 1:
                log.warning("[weather] mode=table: 暦の年 %d は凍結データの範囲外なので"
                            "同月日の最新年(%s)の実測を使う。件数は summary.json の "
                            "weather.days.table_year_remapped に残る。", date.year, rec["date"])
        out = _wg.record_from_observation(rec)
        out["sim_date"] = iso
        if remapped:
            out["source"] = "table_year_remapped"
        return _remember(sim, out)
    # ---- generated ----
    anchor = _wg_anchor_month(sim)
    months = (wcfg["params"] or {}).get("months") or {}
    if str(anchor) not in months:
        _fallback(sim, f"month_{anchor}_not_calibrated")
        return None
    if date.month != anchor:
        # 較正は月ごと。月をまたいだ日に別の月のパラメータを当てるのは較正外の外挿なので
        # やらない(1ランに AR(1) 連鎖は1本=基準月の分だけ)。
        _fallback(sim, "outside_anchor_month")
        return None
    anchor_dom = _calendar.date_of(cal, 0).day
    ser = _generated_series(sim, wcfg, anchor_dom, day_index + 1)
    if day_index >= len(ser):                    # 予定日数を超えた(実質起きない)
        _fallback(sim, "series_exhausted")
        return None
    rec = dict(ser[day_index])
    rec["date"] = date.isoformat()
    _stats(sim)["generated"] += 1
    return _remember(sim, rec)


def _remember(sim, rec: dict) -> dict:
    """実際に使われた非合成の日を控える(summary の要約統計のためだけ。シムは読まない)。"""
    used = getattr(sim, "_weather_used", None)
    if used is None:
        used = []
        sim._weather_used = used
    used.append(rec)
    return rec


def weather_for(sim, day_index: int) -> dict:
    """当日の天気を決定論生成して返す。dict = {"cond", "temp_hi", "temp_lo", …}。

    calendar が enabled のときはその日の月から季節バイアスを掛ける(夏は高温・梅雨は雨増)。
    calendar OFF なら季節なしの穏やかな分布。乱数は新 stream "weather" から1日1回だけ引く。
    mode=generated / table では別経路(前者は "weather_gen" stream・後者は乱数ゼロ)。
    """
    wcfg = sim.weathercfg
    if wcfg.get("mode", "synthetic") != "synthetic":
        rec = _nonsynthetic_for(sim, wcfg, day_index)
        if rec is not None:
            return rec
    rng = sim.hub.stream("weather", day_index)
    cal = getattr(sim, "calendarcfg", None)
    month = None
    if cal and cal.get("enabled"):
        month = _calendar.date_of(cal, day_index * 1440).month
    return _sample(rng, month, wcfg["monthly"], wcfg["neutral"])


def weather_line(weather_dict: dict | None, cfg: dict | None = None) -> str | None:
    """プロンプトへ注入する天気1行。無ければ None(=注入しない=不変)。

    例: "今日の天気: 雨、最高14℃。"
    ★この文言(基底1行)は **1バイトも変えない**(変えると全エージェントの発火/計画/内省
      プロンプトが変わり、過去ランとの比較が全滅する)。拡張(最低気温・湿度・猛暑日・
      暑さ指数)は ``weather.extra_prompt_fields=true`` かつ mode≠synthetic のときだけ
      **末尾に追加**する。語彙は中立(気象庁・環境省の定義語のみ)。
    """
    if not weather_dict:
        return None
    line = f"今日の天気: {weather_dict['cond']}、最高{weather_dict['temp_hi']}℃。"
    if not cfg or not cfg.get("extra_prompt_fields"):
        return line
    if str(cfg.get("mode", "synthetic")) == "synthetic":
        return line                              # 既定モードの文言は絶対に変えない
    return line + _extra_line(weather_dict)


def _extra_line(w: dict) -> str:
    """拡張分の文字列(中立な事実のみ。助言・評価語は入れない)。"""
    parts = []
    if w.get("temp_lo") is not None:
        parts.append(f"最低{w['temp_lo']}℃")
    if w.get("humid") is not None:
        parts.append(f"湿度{w['humid']}%")
    if w.get("precip_mm"):
        parts.append(f"降水{w['precip_mm']}mm")
    out = ("、".join(parts) + "。") if parts else ""
    try:
        if float(w.get("temp_hi_c", w["temp_hi"])) >= _HOT_DAY_C:
            out += "猛暑日(最高気温35℃以上)。"
    except (TypeError, ValueError):               # noqa: BLE001
        pass
    if w.get("wbgt") is not None:
        out += f"暑さ指数(推定){w['wbgt']}。"
    return out


def event_payload(weather_dict: dict, cfg: dict) -> dict:
    """L1 weather イベントの payload。**synthetic では従来どおり2キーのまま**(バイト一致)。

    generated / table のときだけ観測用の列を足す(temp_lo / temp_hi_c / precip_mm /
    humid / wbgt / source)。値が無いキーは足さない(欠測を 0 と偽らない)。
    """
    payload = {"cond": weather_dict["cond"], "temp_hi": weather_dict["temp_hi"]}
    if str(cfg.get("mode", "synthetic")) == "synthetic":
        return payload
    for key in ("temp_lo", "temp_hi_c", "temp_lo_c", "precip_mm", "humid", "wbgt"):
        if weather_dict.get(key) is not None:
            payload[key] = weather_dict[key]
    payload["source"] = weather_dict.get("source", "synthetic")
    if weather_dict.get("source", "").startswith("table"):
        payload["obs_date"] = weather_dict.get("date")   # 実測がどの日のものか(来歴)
    return payload


def provenance(sim) -> dict | None:
    """入力データの来歴 + フォールバック実績(summary.json / run_manifest.json 用)。

    mode=synthetic(既定)では **None**(=既存ランの成果物と同形を保つ)。
    """
    wcfg = getattr(sim, "weathercfg", None) or {}
    if not wcfg.get("enabled") or str(wcfg.get("mode", "synthetic")) == "synthetic":
        return None
    out = dict(wcfg.get("provenance") or {"mode": wcfg.get("mode")})
    st = getattr(sim, "_weather_stats", None)
    if st:
        out["days"] = {"generated": st["generated"], "table": st["table"],
                       "table_year_remapped": st.get("table_year_remapped", 0),
                       "synthetic_fallback": st["synthetic_fallback"],
                       "fallback_reasons": dict(st["fallback_reasons"])}
    used = getattr(sim, "_weather_used", None)
    if used:
        from . import weather_gen as _wg
        out["series"] = _wg.series_stats(used)      # 実際にランで使われた日だけを要約する
        if getattr(sim, "_weather_series_clipped", None) is not None:
            out["series"]["n_clipped_in_generated_series"] = int(sim._weather_series_clipped)
    return out


def discomfort_delta(weather_dict: dict | None, cfg: dict) -> float:
    """悪天候(雨・雪)の日の不快感の量(不透明 magnitude)。scheduler が factors へ渡す。

    晴・曇や rain_grievance=0.0(既定)のときは 0.0(=加算なし)。乱数は引かない(決定論)。
    """
    if weather_dict and weather_dict.get("cond") in _BAD_CONDS:
        return float(cfg.get("rain_grievance", 0.0))
    return 0.0
