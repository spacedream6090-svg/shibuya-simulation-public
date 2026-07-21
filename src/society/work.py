"""L2 域内従業者の「業務の実体」(work.service。既定 OFF)。

ユーザー要望 2026-07-20:「L2 の人々も接客などのサービスを行っている、もしくは会社単位で
何かのサービスを作っているだろうからそれを反映したい」。実査は docs/research/l2-work-reality.md。

勤務中(work_node に在場・勤務時間帯)のエージェントへ「業務の実体」を与える決定論機構:
  - 接客系: 客の消費 spend(食事/カフェ/ナイトライフ/買い物)と同一 work_node に居る勤務中スタッフに
    serve を帰属(機械的=LLM 呼ゼロ・乱数ゼロ)。客側の既存イベントは不変(新イベントを足すだけ)。
  - オフィス系: 日次境界で職場単位に「出勤者数 × role重み」を org_output として集計(会社が
    「何かを作っている」の最小観測形)。

原則(R1 doctrine):
- **既定 OFF**(service.enabled=false)。OFF 時は本モジュールの経路を一切通さない=バイト一致。
- 乱数を一切引かない(新 stream 不要)・LLM 呼び出しを一切増やさない。判定は observables
  (work_node/在場/勤務時間帯/POI カテゴリ/occupation)のみ=k(信念書き戻し)非依存。
- 業種名・業務名テキストはここ(本モジュール既定)/config 由来に閉じる。本モジュールは
  src/society 直下(engine/cognition/actions/labeling/world/factors の禁則ディレクトリ外)。

配線: engine/simulation.py が `sim.workcfg = work.build_cfg(cfg.get("work"))` を1回だけ正準化して保持し、
engine/scheduler.py の `_phase_work_service` が run_step 末で本モジュールの純関数を使う。
"""
from __future__ import annotations

# 客の消費カテゴリ(spend.cat)→ 接客業務ラベルの既定写像(config 未指定時)。
# spend.cat は economy/scheduler 由来: 食事/ナイトライフ=food/nightlife、買い物=shop、
# P2 buy の leisure=cafe。taxi/bus(交通)は接客対象外なので写像に入れない。
DEFAULT_SERVE_BY_CAT = {
    "food": "飲食の接客",
    "cafe": "飲食の接客",
    "nightlife": "接客・サービス",
    "shop": "販売・レジ対応",
}

# ダイジェストで「業務が多かった」と言い換える件数の閾値(客観記述の粒度)。
_MANY_THRESHOLD = 8


def build_cfg(raw: dict | None) -> dict:
    """conf の work ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    svc = dict(raw.get("service", {}) or {})
    serve = svc.get("serve_by_cat")
    serve = dict(serve) if serve else dict(DEFAULT_SERVE_BY_CAT)
    off = dict(svc.get("office", {}) or {})
    return {
        "enabled": bool(svc.get("enabled", False)),
        # 客の消費カテゴリ → 接客ラベル(業務名テキストは config 由来)。
        "serve_by_cat": {str(k): str(v) for k, v in serve.items()},
        # 1消費あたり帰属するスタッフ数の上限(id 昇順で先頭から)。
        "max_serve_per_event": max(0, int(svc.get("max_serve_per_event", 1))),
        # スタッフ不在の消費を agent_id=-1 の記録として残すか(挙動変更なし・観測のみ)。
        "record_unstaffed": bool(svc.get("record_unstaffed", True)),
        # interstitial(S2)ダイジェストへ業務要約を1行供給するか(ON 時のみ効く)。
        "digest": bool(svc.get("digest", True)),
        "office": {
            "enabled": bool(off.get("enabled", True)),
            # オフィス系職場と見なす職場 POI カテゴリ(既定 office)。
            "poi_cats": [str(c) for c in (off.get("poi_cats") or ["office"])],
            "base_weight": float(off.get("base_weight", 1.0)),
            # occupation/role → 産出重み(空=全員 base_weight=出勤者数に比例)。
            "role_weights": {str(k): float(v)
                             for k, v in (off.get("role_weights") or {}).items()},
        },
    }


def enabled(cfg: dict | None) -> bool:
    return bool(cfg and cfg.get("enabled"))


# ---------------------------------------------------------------- 接客(serve)
def serve_label(cfg: dict, cat: str | None) -> str | None:
    """消費カテゴリ cat の接客ラベル(接客対象外=None)。"""
    if not cat:
        return None
    return cfg["serve_by_cat"].get(str(cat))


def note_serve(agent, label: str) -> None:
    """スタッフ本人の当日業務アキュムレータへ1件加える(ダイジェスト供給用)。

    属性は必要時にのみ生やす(OFF/非該当は属性不在=interstitial の実行時状態を汚さない)。"""
    acc = getattr(agent, "_work_serve_by_label", None)
    if acc is None:
        acc = {}
        agent._work_serve_by_label = acc
    acc[label] = acc.get(label, 0) + 1


def clear_digest(agent) -> None:
    """発火(ダイジェスト取得)時に当日業務アキュムレータを空にする(前回発火以降の仕切り直し)。"""
    if getattr(agent, "_work_serve_by_label", None):
        agent._work_serve_by_label = {}


def digest_line(agent) -> str | None:
    """スタッフの当日業務を interstitial ダイジェストの1事実に整形(客観記述・意味づけしない)。

    アキュムレータ不在/空なら None(=1行も足さない=バイト一致)。テキストは本モジュールに閉じる。"""
    acc = getattr(agent, "_work_serve_by_label", None)
    if not acc:
        return None
    total = sum(acc.values())
    top = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    if total >= _MANY_THRESHOLD:
        return f"今日は{top}が多かった(業務{total}件)"
    return f"今日は{top}などの業務をこなした(業務{total}件)"


# ---------------------------------------------------------------- オフィス産出(org_output)
def is_office_node(city, node: str, ocfg: dict) -> bool:
    """work_node がオフィス系(poi_cats のいずれかの POI を持つ)か(決定論・乱数なし)。"""
    cats = ocfg["poi_cats"]
    for p in city.pois_at_node(node):
        if p.get("cat") in cats:
            return True
    return False


def role_weight(agent, cfg: dict) -> float:
    """出勤者の産出重み。occupation/role → role_weights、無ければ base_weight(=1人1.0)。"""
    ocfg = cfg["office"]
    key = getattr(agent, "org_role", "") or getattr(agent, "occupation", "")
    return ocfg["role_weights"].get(str(key), ocfg["base_weight"])
