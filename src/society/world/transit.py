"""公共交通(定刻ダイヤ)。ダイヤ乱れなし=ファイルの時刻どおり決定論的に運行。

現状のダイヤは公表の始発・終電・運転間隔から生成した近似(ファイルに明記)。
★実ダイヤへの切替(v3): conf の transit.gtfs_dir に GTFS(stops.txt / stop_times.txt /
trips.txt / routes.txt)を置いたフォルダを指定すると、対象駅の実発車時刻から
路線ごとの始発・終電・間隔を再構成して置き換える。対象駅名は envpack の
station_filters(場所の値)で与える(基盤に駅名を持たせない)。
GTFS の入手 = ODPT(公共交通オープンデータセンター)。developer.odpt.org で無料の
開発者登録(確認に最大2営業日)→ アクセストークン発行 → ckan.odpt.org から
事業者別 GTFS を取得。
役割: (1) 駅経由の退出/帰還を運行時間帯に制約(終電後は帰って来られない)
      (2) 到着ごとの人の波(パルス)の源。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .clock import STEP_MINUTES


def _to_min(text: str) -> int:
    parts = text.split(":")
    return int(parts[0]) * 60 + int(parts[1])


# 居住沿線ラベル → 路線名 の照合に要する最小の共通部分文字列長。
# 名簿の residence_line は**居住エリアのラベル**(「二子玉川・田園都市線沿線」「杉並方面」)で
# あって路線名ではない。路線名の核(「山手線」「東横線」「井の頭線」「田園都市線」)はこの
# データでは 3 文字以上あり、2 文字以下の一致は事業者接頭辞(東京/東急/京王)や地名の
# 偶然の重なりでしかない。誤った路線へ結ぶより**合併集合へ落とす**ほうが安全なので、
# 閾値は保守側の 3 に置く(下げると偶然一致で別路線のダイヤに乗る)。
_LINE_MATCH_MIN = 3


def _lcs_len(a: str, b: str) -> int:
    """最長共通**部分文字列**の長さ(連続。純関数・地名リテラルを 1 つも持たない)。"""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _load_gtfs(gtfs_dir: Path, station_filters: list[str]) -> list[dict] | None:
    """GTFS から対象駅停車の路線別 始発/終電/中央間隔を再構成する(最小実装)。

    station_filters は対象駅名の部分一致リスト(envpack=場所の値)。空なら抽出不能=None。
    """
    if not station_filters:
        return None
    try:
        stops = list(csv.DictReader(
            (gtfs_dir / "stops.txt").open(encoding="utf-8-sig")))
        station_ids = {s["stop_id"] for s in stops
                       if any(f in s.get("stop_name", "") for f in station_filters)}
        if not station_ids:
            return None
        trips = {t["trip_id"]: t["route_id"] for t in csv.DictReader(
            (gtfs_dir / "trips.txt").open(encoding="utf-8-sig"))}
        routes = {r["route_id"]: (r.get("route_long_name")
                                  or r.get("route_short_name") or r["route_id"])
                  for r in csv.DictReader(
                      (gtfs_dir / "routes.txt").open(encoding="utf-8-sig"))}
        deps: dict[str, list[int]] = {}
        for st in csv.DictReader(
                (gtfs_dir / "stop_times.txt").open(encoding="utf-8-sig")):
            if st["stop_id"] in station_ids and st.get("departure_time"):
                route = routes.get(trips.get(st["trip_id"], ""), "不明")
                deps.setdefault(route, []).append(_to_min(st["departure_time"]))
        lines = []
        for name, times in sorted(deps.items()):
            times.sort()
            gaps = sorted(b - a for a, b in zip(times, times[1:]) if 0 < b - a < 60)
            headway = gaps[len(gaps) // 2] if gaps else 5
            first, last = times[0] % 1440, times[-1]
            lines.append({"name": name, "first": f"{first // 60}:{first % 60:02d}",
                          "last": f"{last // 60}:{last % 60:02d}",
                          "headway_min": headway,
                          # 3b: 1 本ごとの実発車時刻を**畳まずに**持つ(transit.real_departures
                          #     が ON のときだけ arrivals_between がこちらを使う)。
                          "departures": list(times),
                          "_first": first, "_last": last % 2880,
                          "source": "GTFS(実ダイヤ)"})
        return lines or None
    except (OSError, KeyError, ValueError):
        return None


def _canon_departures(raw) -> tuple[int, ...]:
    """``departures``(1 本ごとの実発車時刻。サービス日の分)を正準化する純関数。

    ★正準化の中身は 3 つだけ: int 化・重複除去・昇順。負値と 2 日ぶん(2880 分)を超える値は
      捨てる(サービス日は「その日の始発〜翌 2 時台」= 最大でも 1440+180 なので、それを
      超える値はデータの壊れ)。列が無い/空なら空タプル = 従来の等間隔再構成へ後退する。
    """
    if not raw:
        return ()
    out: set[int] = set()
    for v in raw:
        try:
            m = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= m < 2880:
            out.add(m)
    return tuple(sorted(out))


def snap_to_arrival_of_day(table, min_of_day: int):
    """到着表(**昇順**の ``(分 of day, 路線名)``)から「その分以上で最も早い到着」。

    ★**読み取り専用の純関数**(状態を 1 バイトも変えない・乱数もファイル I/O も無い)。
    ``engine/simulation._snap_to_arrival`` と**同じ 3 規則**を「分 of day」の領域で持つ
    (あちらは絶対 sim 分の領域):

      (1) 通常   … 与えられた分**以上**で最も早い到着へ
      (2) 始発前 … 表の先頭(= その日の始発)へ ((1) が自然に満たす)
      (3) 終電後 … 表の末尾(= その日の最終到着)へクランプ

    同着(同じ分に複数路線)は表の並びどおり = **路線名の昇順**が先に来る
    (``arrivals_between`` が ``(時刻, 路線名)`` で sort しているため)= 一意。
    表が空なら ``None``(捏造しない)。
    """
    if not table:
        return None
    m = int(min_of_day) % 1440
    for t, name in table:
        if int(t) >= m:
            return (int(t), str(name))
    return (int(table[-1][0]), str(table[-1][1]))


class Transit:
    def __init__(self, path: str | Path, gtfs_dir: str | Path | None = None,
                 station_filters: list[str] | None = None,
                 real_departures: bool = False):
        # 対象駅名(GTFS 抽出用)は envpack=場所の値。既定なし=GTFS 実ダイヤ切替時のみ要る。
        self.station_filters = list(station_filters or [])
        # 第114 レーン 3b(既定 False=従来の等間隔再構成のまま=バイト一致):
        #   True かつ路線が `departures`(1 本ごとの実発車時刻)を持つなら、その列を
        #   そのまま到着表に使う。持たない路線は従来どおり (始発, 間隔, 終電) の再構成。
        self.real_departures = bool(real_departures)
        self.source = "approximation"
        # 運休フラグ(都市・環境ショック 後続波 H4。既定 False=通常運行)。災害/運休の日に
        # disaster 層が True にすると駅経由の退出/帰還が不可(交通麻痺)。disaster OFF では常に
        # False のまま=has_service は現行挙動とバイト一致。
        self.suspended = False
        # 簡易バスの路線定義(ユーザー要望 2026-07-06)。既定 OFF なので通常は休眠。
        # ファイルの "bus_lines"(中立名の停留所ノード列)を素の list として保持する。
        raw_file = json.loads(Path(path).read_text(encoding="utf-8"))
        self.bus_lines: list[dict] = list(raw_file.get("bus_lines", []))
        lines = None
        if gtfs_dir:
            lines = _load_gtfs(Path(gtfs_dir), self.station_filters)
            if lines:
                self.source = "gtfs"
        if lines is None:
            lines = raw_file["lines"]
            for line in lines:
                line["_first"] = _to_min(line["first"])
                line["_last"] = _to_min(line["last"])  # 24:30 → 1470(翌日 0:30)
        for line in lines:                             # 3b: 実発車時刻列の正準化
            line["_departures"] = _canon_departures(line.get("departures"))
        self.lines: list[dict] = lines

    def lines_in_service(self, sim_min: int) -> list[str]:
        """この時刻(シミュ内分)に運行中の路線名。10分 step ≧ 最短間隔なので在/不在で十分。"""
        mod = sim_min % 1440
        names = []
        for line in self.lines:
            first, last = line["_first"], line["_last"]
            in_service = (first <= mod <= last) if last < 1440 else \
                         (mod >= first or mod <= last - 1440)
            if in_service:
                names.append(line["name"])
        return names

    def has_service(self, sim_min: int) -> bool:
        if self.suspended:                     # 運休(災害/運休の日。既定 False=通常運行=不変)
            return False
        return bool(self.lines_in_service(sim_min))

    # ------------------------------------------------------------------ #
    # 到着時刻の列挙(読み取り専用の純関数。状態を 1 バイトも変えない)
    # ------------------------------------------------------------------ #
    def arrivals_between(self, min_a: int, min_b: int) -> list[tuple[int, str]]:
        """[min_a, min_b] (両端含む・**絶対** sim 分)の到着を (sim_min, 路線名) で返す。

        ★分解能(第114 レーン 3b で**路線ごとに**上がった。以下は正直な現状)
        ------------------------------------------------------------------
        路線が ``departures``(1 本ごとの実発車時刻の列)を持ち、かつ
        ``transit.real_departures`` が ON なら**その列をそのまま**返す(実ダイヤのコピー)。
        持たない路線は従来どおり (始発, 終電, 運転間隔) の 3 値からの**等間隔の再構成**で、
        これは実ダイヤのコピーではない。混在するのが普通で、どちらなのかは路線の
        ``source`` 欄で事後に判る:

          - 実ダイヤ … 事業者の駅時刻表 / 静的 GTFS が取れた路線(同梱の実ダイヤでは 6/9 路線)。
            朝ラッシュの短縮間隔がそのまま入る。
          - 近似 … 駅時刻表が非提供の路線(同 3 路線)。公表の始発/終電/日中間隔から
            等間隔に刻む。朝ラッシュの短縮は入っていない。

        どちらも完全な決定論(乱数ゼロ・呼ぶ順に依存しない)。並びは (時刻, 路線名) の
        辞書順で、同一分に複数路線が着くときは**路線名の昇順**が先頭に来る。

        日跨ぎ: 終電は 24:36 のように 1440 を超える値で保持されている(サービス日の
        表現)。前日のサービス日から溢れた深夜の到着も拾うため走査は 1 日前から始める。
        """
        a, b = int(min_a), int(min_b)
        if b < a:
            return []
        out: list[tuple[int, str]] = []
        for day in range(a // 1440 - 1, b // 1440 + 1):
            base = day * 1440
            for line in self.lines:
                name = str(line["name"])
                deps = line.get("_departures") or ()
                if self.real_departures and deps:      # 3b: 実発車時刻をそのまま使う
                    for t in deps:
                        s = base + int(t)
                        if a <= s <= b:
                            out.append((s, name))
                    continue
                first, last = int(line["_first"]), int(line["_last"])
                headway = max(1, int(line.get("headway_min", 1) or 1))
                t = first
                while t <= last:
                    s = base + t
                    if a <= s <= b:
                        out.append((s, name))
                    t += headway
        out.sort()
        return out

    def line_for_residence(self, label: str) -> str | None:
        """居住沿線ラベル → **積んでいるダイヤの路線名** / None(該当なし)。読み取り専用。

        ★照合規則を正直に(ここを誤解すると『沿線別のダイヤ』が嘘になる)
        ------------------------------------------------------------
        名簿の `residence_line` は envpack の `residence_lines`(居住エリアのラベル)で、
        **路線名とは別の語彙**である。実データでは
          「二子玉川・田園都市線沿線」「横浜・東横線沿線」「下北沢・井の頭線沿線」
          「新宿・山手線沿線」… = 路線名を**含む**ラベル
          「杉並方面」「世田谷方面」「目黒方面」「川崎・多摩方面」= 路線を名指さないラベル
        の 2 種が混在する。**完全一致は 1 件も成立しない**(exact match 率 0%)。

        そこで規則は「最長共通**部分文字列**が `_LINE_MATCH_MIN` 文字以上なら同じ路線」。
        同点は (一致長の降順, 路線名の昇順) で解く = 完全な決定論。「新宿・山手線沿線」は
        内回り/外回りの両方に 3 文字一致するので**路線名の昇順で内回り**に落ちる
        (同じ駅の同じ改札に着くので、どちらを採っても到着の物理は変わらない)。
        該当なし(= 路線を名指さない居住ラベル)は None を返し、呼び出し側は
        **全路線の合併集合**へ落とす(誤った路線へ結ぶより安全側)。

        地名・路線名のリテラルはコード側に 1 つも持たない(比較するのは名簿の値と
        ダイヤの値だけ)= envpack ドクトリンに従う。
        """
        text = str(label or "")
        if not text:
            return None
        # (一致長の降順, 路線名の昇順)= sort 1 発で同点解決まで決まる
        scored = sorted((-_lcs_len(text, str(ln["name"])), str(ln["name"]))
                        for ln in self.lines)
        if not scored:
            return None
        best_n, best_name = -scored[0][0], scored[0][1]
        return best_name if best_n >= _LINE_MATCH_MIN else None

    def arrivals_of_day(self) -> list[tuple[int, str]]:
        """定常日の到着を **分 of day** [0, 1440) で列挙する(arrivals_between の薄い別名)。

        24:36 の終電は翌 0:36 の到着として 36 分の位置に現れる(前日サービス日からの
        折り返しを arrivals_between が拾う)= 分 of day の領域で閉じた集合になる。
        """
        return self.arrivals_between(0, 1439)

    def arrival_at_or_after(self, min_of_day: int):
        """分 of day 以上で最も早い到着 ``(分 of day, 路線名)`` / ``None``(ダイヤ空)。

        ``arrivals_of_day`` + ``snap_to_arrival_of_day`` の薄い別名(**読むだけ**)。
        毎 step 呼ぶ用途では呼び出し側が ``arrivals_of_day()`` を 1 回だけ組んで
        ``snap_to_arrival_of_day`` を直接使うこと(本 method は表を毎回組み直す)。
        """
        return snap_to_arrival_of_day(self.arrivals_of_day(), int(min_of_day))


class BusNetwork:
    """簡易バス(ユーザー要望 2026-07-06)。既定 OFF = 本番トグル。

    電車の車内・駅構内の高度化はしない(GTFS 実ダイヤ API は申請待ち)。ここは
    「同一路線の停留所が近い出発/目的の間を、便(headway)ごとに短い区間乗車で結ぶ」
    最小実装。has_service 流儀を踏襲し serves() で便の有無を、find_ride() で乗車可否を
    決定論的に返す(乱数なし)。停留所は node id か {node:...} で与え、地図に存在する
    ノードだけを採用する(合成データ・簡易 JSON の両方に対応)。"""

    def __init__(self, lines, city, stop_radius_m: float = 100.0,
                 headway_steps: int = 1, step_minutes: int = STEP_MINUTES):
        self.radius = float(stop_radius_m)
        self.headway = max(1, int(headway_steps))
        # 中央 Δt(A3・第94バッチ OBS-U2)。serves() は sim_min を step 番号へ戻すので
        # Δt を知る必要がある。既定 10 = 従来の直書きと厳密同値(traffic.py と同じ配線)。
        self.step_minutes = max(1, int(step_minutes))
        self.lines: list[dict] = []
        for ln in (lines or []):
            stops: list[tuple[str, float, float]] = []
            for s in ln.get("stops", []):
                node = s if isinstance(s, str) else \
                    (s.get("node") if isinstance(s, dict) else None)
                if node is not None and node in city.graph.nodes:
                    x, y = city.node_xy(node)
                    stops.append((node, x, y))
            if len(stops) >= 2:
                self.lines.append({"name": ln.get("name", "循環バス"),
                                   "stops": stops})

    def serves(self, sim_min: int) -> bool:
        """この step に便があるか(headway ごと)。既定 headway=1 step なら常に True。

        A3: 分 → step は Δt に依存する(直書き 10 のままだと Δt=1 で同じ便判定が 10 step
        続き、headway が実質 10 倍になる)。headway_steps 自体は timeconv の STEPS 類なので
        config 側で ×10 され、ここが step_minutes で割れば実時間の便間隔が保存される。"""
        return ((sim_min // self.step_minutes) % self.headway) == 0

    def _nearest(self, x: float, y: float, stops):
        best, best_d = None, self.radius
        for node, sx, sy in stops:
            d = ((sx - x) ** 2 + (sy - y) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = node, d
        return best

    def find_ride(self, from_node: str, to_node: str, city):
        """出発・目的の両方が同一路線の停留所の近く(<radius)なら {line, from, to} を返す。"""
        if from_node not in city.graph.nodes or to_node not in city.graph.nodes:
            return None
        fx, fy = city.node_xy(from_node)
        tx, ty = city.node_xy(to_node)
        for ln in self.lines:
            a = self._nearest(fx, fy, ln["stops"])
            b = self._nearest(tx, ty, ln["stops"])
            if a is not None and b is not None and a != b:
                return {"line": ln["name"], "from": a, "to": b}
        return None
