"""建物の実階数への floor クランプ 3D-U0(``world.floor_clamp``・**既定 OFF**)。

正典
----
- ``PENDING.md`` 3D-U0「sim 側 floor クランプ(建物階数超の floor が通る = L1 が変わる
  修正。表示側は修正済み)」
- ``docs/plans/highfidelity-3d-physics-plan.md`` §1 / §4(ユーザー判断 3D-U0。
  「conf トグル既定 OFF で実装し観察ランで ON」が推奨)
- ``docs/log/devlog-block15-fulltext.md`` Entry 89(実査: 2 階建てに floor=42 →
  +138m 浮遊・描画フレームの 5.4% が屋根超え)

何を解く問題か
--------------
POI データ(``data/shibuya_osm.json``)には **建物の階数 levels を超える floor** を持つ
POI が 20 件、**地下(負の floor)** を持つ POI が 31 件ある。職場・バイト先の階は
この POI の値から来るため、``agent.floor`` に建物階数を超える値がそのまま入り、
L1 の ``enter_building`` payload に記録される(実測: mock 40 体 24 step で 1 件)。

**表示側は 2026-08-02 に修正済み**(``scripts/export_3d.py::encode_indoor_w``)。
本 module は **その規則を 1 バイト単位で写した sim 側**である。「シムとビューアで
別の値になる」ことが原理的に起きないよう、純関数 :func:`clamp_floor` の出力が
``encode_indoor_w`` の階部分と一致することを ``tests/test_floor_clamp.py`` が
格子上で機械固定する。

規則(``encode_indoor_w`` と同一)
----------------------------------
``floor`` を ``[1, max(1, min(levels, 99))]`` へクランプする。

- 0・負値 → **1**(地上 1 階)。``floor == 0`` は sim では「建物の外」の規約なので、
  屋内位置としての 0 は存在しない = クランプ対象は「屋内に居るときの階」だけ。
- ``levels`` 超え → ``levels``。``levels`` が 0/None → 1。
- 上限 99 は表示側 w = ``1000 + bIdx*100 + floor`` の 2 桁枠(超えると **別の建物**を
  指してしまう)に由来する。実データの最大 levels は 47 なので実効差は無いが、
  「同一の規則」を名乗る以上そろえる。

**地下(below)は扱わない。** 建物データには ``below``(最大 7)があり
``viz/make_viewer.py`` の階チップ列挙だけが -1..-below を出すが、位置エンコード w は
1..99 しか表現できず、sim も ``floor == 0`` を屋外に使っているため地下階を表現できない。
現状すでに ``persona._pick_workplace`` / ``economy.assign_part_time`` / ``work.py`` が
``max(1, floor)`` で地下 POI を 1F へ潰している = **sim もビューアも「地下は 1F」で一致**。
地下階の導入は L1 の意味論変更であって 3D-U0 の範囲外。

何をクランプし、何をクランプしないか
--------------------------------------
クランプするのは **位置 ``agent.floor`` だけ**。``work_floor`` / ``home_floor`` /
``part_time["floor"]`` は「名簿・台帳の上での職場の階(POI 由来)」であって位置ではない
ので触らない(源を書き換えると名簿 parquet と POI データの整合が崩れ、データ側の
逸脱が観測できなくなる)。建物へ入る経路は必ず

  ``enter_building`` / ``floor_move`` / ``go_to_bed`` / natural_start の着席

の 4 つを通るので、位置としての逸脱はその 4 代入点の :func:`clamp` で閉じる。
``exit_building`` 等の ``agent.floor = 0``(屋外)と lodging の ``= 1`` は定義上
妥当なので通さない。

既存の常時 ON クランプとの関係
--------------------------------
``mobility._set_home``(住居の割当)は ``max(1, min(floor, levels))`` を **トグル無しで**
実施済み = 本件以前からの正実装。99 上限が無い点だけ本 module と違うが、実データの
最大 levels は 47 で実効差はゼロ。**既定 OFF のバイト一致を最優先し、そちらは触らない**。

R1
--
決定論(乱数を 1 つも引かない純関数)・新規 stream ゼロ・既存 draw 順不変。
既定 OFF では :func:`clamp` は受け取った値をそのまま返す = L1 バイト一致。
"""
from __future__ import annotations

# 表示側 w = 1000 + bIdx*100 + floor の 2 桁枠(scripts/export_3d.W_FLOOR_MAX と同値)
W_FLOOR_MAX = 99


def clamp_floor(floor, levels) -> int:
    """``floor`` を ``[1, max(1, min(levels, 99))]`` へ丸める(純関数・乱数ゼロ)。

    ``scripts/export_3d.py::encode_indoor_w`` の階部分と**同一の規則**。
    int にできない値は 1(表示側と同じ退避)。
    """
    try:
        f = int(floor)
    except (TypeError, ValueError):
        f = 1
    hi = max(1, min(int(levels or 1), W_FLOOR_MAX))
    return max(1, min(f, hi))


def enabled(sim) -> bool:
    """``world.floor_clamp.enabled``(初回のみ ``sim.cfg`` から読んでキャッシュ)。

    ``traces.cfg_of`` / ``rumors.cfg_of`` と同型の遅延読み(``simulation.py`` の
    初期化配線をゼロタッチに保つ)。キャッシュ属性 ``sim._floor_clamp_on`` は
    L1/L2/L3/乱数のいずれにも現れない = 既定 OFF のバイト一致を壊さない。
    """
    on = getattr(sim, "_floor_clamp_on", None)
    if on is None:
        try:
            raw = (sim.cfg.get("world", None) or {}).get("floor_clamp", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        on = bool((raw or {}).get("enabled", False))
        sim._floor_clamp_on = on
    return on


def clamp(sim, building, floor):
    """屋内位置の階を建物の実階数へ丸める(**OFF なら受け取った値をそのまま返す**)。

    ``building`` は建物 dict でも建物 id でもよい(呼び出し側に dict があるときは
    余計な引き直しをしない)。建物が解決できないときはクランプしない
    (存在しない建物の階数を捏造しない = 表示側が w=0 へ退避するのと同じ線引き)。
    """
    if not enabled(sim):
        return floor
    bld = building
    if not isinstance(bld, dict):
        city = getattr(sim, "city", None)
        if city is None or not bld or not city.has_building(bld):
            return floor
        bld = city.building(bld)
    return clamp_floor(floor, bld.get("levels"))
