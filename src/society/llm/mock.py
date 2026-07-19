"""Mock LLM(GPU 不要)。決定論的に「それらしい」JSON 行動を返す。

役割: P0 骨格の配線検証。プロンプト内容 + rng_key だけから応答が決まる
(呼び出し順序に依存しない)ので、決定論リプレイのテストにも使える。
新語の造語もここで行う(LLM の代役として。engine は関与しない=指紋なし)。
"""
from __future__ import annotations

import json

from ..rng import RngHub
from .base import LLMBackend

_SYLLABLES = ["モヤ", "ザワ", "グラ", "ピカ", "ドタ", "フワ", "ギュウ", "スカ",
              "ヌル", "バタ", "シブ", "ガヤ", "ポカ", "ジリ", "ワサ"]
_SUFFIXES = ["り現象", "み", "き", "ら化", "つき", "ん街", "スポット", "族"]

_SMALLTALK = [
    "{place}、今日はずいぶん人が多いね。",
    "{place}のあたり、最近ちょっと雰囲気変わった気がする。",
    "ここ、待ち合わせにはいいんだけど混みすぎなんだよな。",
    "{place}でこの時間は珍しく静かだ。",
    "またここ通るのか。近道がないんだよね。",
]
_COMPLAINTS = [
    "正直、{place}はもう歩きにくすぎる。どうにかならないのか。",
    "{place}の混雑、誰か何とかしてほしいよ。",
    "この街、人が増えるばかりで居場所が減ってる気がする。",
]


class MockBackend(LLMBackend):
    name = "mock"

    def __init__(self, hub: RngHub):
        self.hub = hub

    def _coin_word(self, rng) -> str:
        parts = "".join(_SYLLABLES[int(rng.integers(len(_SYLLABLES)))]
                        for _ in range(int(rng.integers(1, 3))))
        return parts + _SUFFIXES[int(rng.integers(len(_SUFFIXES)))]

    def _mock_rule(self, rng, place: str, nearby: list) -> dict:
        """制度DSL の機械可読ルールを4型巡回で1つ生成する(すべて検証を通る妥当値)。"""
        rtype = ["fee", "bonus", "curfew", "weekly_event"][int(rng.integers(4))]
        if rtype == "fee":
            cat = ["food", "cafe", "shop", "nightlife"][int(rng.integers(4))]
            delta = int(rng.choice([-200, -100, 100, 200]))
            return {"type": "fee", "target_cat": cat, "delta": delta}
        if rtype == "bonus":
            beh = ["park", "event_attend", "flyer_view"][int(rng.integers(3))]
            return {"type": "bonus", "behavior": beh,
                    "amount": int(rng.integers(1, 6)) * 100}      # 100〜500 円
        if rtype == "curfew":
            cat = ["nightlife", "leisure", "shop"][int(rng.integers(3))]
            return {"type": "curfew", "target_cat": cat, "from_h": 22, "to_h": 5,
                    "weight": 0.3}
        place_name = nearby[int(rng.integers(len(nearby)))] if nearby else place
        return {"type": "weekly_event", "title": f"{place}の朝市",
                "place": place_name, "every_days": int(rng.integers(1, 8))}

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        # think は mock では無視(実LLMの思考モードは応答本文には影響しない前提)。
        # プロンプト全文で乱数を引く(信念など後方の変化も応答に反映=実LLMの近似)
        rng = self.hub.stream("mock", rng_key, prompt)
        place = "この辺"
        for line in prompt.splitlines():
            if line.startswith("場所:"):
                place = line.split(":", 1)[1].strip()
        known_terms = []
        for line in prompt.splitlines():
            if line.startswith("知っている言葉:"):
                raw = line.split(":", 1)[1].strip()
                known_terms = [t for t in raw.split("、") if t]

        # 日課計画フレームワーク(P2 S1・planning.framework=true のときだけ prompt に
        # "計画スキーマ:" マーカーが載る)。スキーマ準拠の決定論応答を返す=mock でも
        # コンパイラが通る。OFF ではこのマーカーが無い=下の従来分岐へ=バイト一致。
        if "計画スキーマ:" in prompt:
            nearby = []
            for line in prompt.splitlines():
                if line.startswith("周りにある店・場所:"):
                    nearby = [t for t in line.split(":", 1)[1].strip().split("、")
                              if t]
            bands = ["朝", "昼", "午後", "夕方", "夜"]
            # mandatory(勤務・就学)はアンカー系統=シフト窓が担うので mock は出さない。
            maint = ["shop", "personal", "escort", "meal"]
            discr = ["leisure", "park", "walk", "visit"]
            n = int(rng.integers(2, 5))                    # 2-4 個
            picks = sorted(int(i) for i in
                           rng.choice(len(bands), size=n, replace=False))
            items = []
            for bi in picks:
                if rng.random() < 0.5:
                    cat, what = "maintenance", maint[int(rng.integers(len(maint)))]
                else:
                    cat, what = "discretionary", discr[int(rng.integers(len(discr)))]
                where = ""
                if nearby and rng.random() < 0.5:
                    where = nearby[int(rng.integers(len(nearby)))]
                elif place != "この辺" and rng.random() < 0.4:
                    where = place
                band = bands[bi]
                items.append({"intent": f"{band}に{what}をする", "cat": cat,
                              "what": what, "place": where, "when": band,
                              "flex": "fixed" if cat == "maintenance" else "flexible"})
            return json.dumps({"action": "plan", "items": items},
                              ensure_ascii=False)

        # 朝の一日計画(planning.make_plan の計画タスク)。routine の土台になる予定を
        # rng で 2-4 個。時間帯は重複なし、内容はカテゴリからランダム、場所は周辺 POI や
        # 現在地名を一部に採用(place 部分一致の解決を機構として動かすため)。
        if "今日一日の計画" in prompt:
            nearby = []
            for line in prompt.splitlines():
                if line.startswith("周りにある店・場所:"):
                    nearby = [t for t in line.split(":", 1)[1].strip().split("、")
                              if t]
            bands = ["朝", "昼", "午後", "夕方", "夜"]
            whats = ["work", "meal", "shop", "leisure", "park", "walk", "home",
                     "visit"]
            n = int(rng.integers(2, 5))                    # 2-4 個
            picks = sorted(int(i) for i in
                           rng.choice(len(bands), size=n, replace=False))
            items = []
            for bi in picks:
                what = whats[int(rng.integers(len(whats)))]
                where = ""
                if nearby and rng.random() < 0.5:
                    where = nearby[int(rng.integers(len(nearby)))]
                elif place != "この辺" and rng.random() < 0.4:
                    where = place
                items.append({"when": bands[bi], "what": what, "place": where})
            return json.dumps({"action": "plan", "items": items},
                              ensure_ascii=False)

        surprised = "驚き:" in prompt
        roll = rng.random()

        # ツール(中立提示。低確率で自然に使う=研究対象の観察。促進はしない)。
        # 提示行があるとき(solo/social/post 発火)だけ、各ツールを 2-5% で返す分岐。
        tline = next((ln for ln in prompt.splitlines()
                      if ln.startswith("他にできること")), None)
        if tline is not None:
            t = rng.random()
            if "open_venture" in tline and t < 0.03:
                return json.dumps(
                    {"action": "open_venture", "name": f"{place}の屋台",
                     "offer": "コーヒーとおにぎり"}, ensure_ascii=False)
            if 0.03 <= t < 0.06:
                term = (known_terms[int(rng.integers(len(known_terms)))]
                        if known_terms else None)
                title = f"{term}を語る会" if term else f"{place}で集まろう"
                return json.dumps(
                    {"action": "host_event", "title": title,
                     "hours_later": int(rng.integers(1, 4))}, ensure_ascii=False)
            if 0.06 <= t < 0.09:
                return json.dumps(
                    {"action": "post_flyer",
                     "text": f"{place}で気になることがある。よかったら見て。"},
                    ensure_ascii=False)
            if 0.09 <= t < 0.11:
                return json.dumps(
                    {"action": "found_group", "name": f"{place}の会",
                     "purpose": "この辺の人でゆるくつながる"}, ensure_ascii=False)
            if 0.11 <= t < 0.13:
                prop = {"action": "propose",
                        "text": f"{place}の混雑をどうにかしよう"}
                if '"rule"' in tline:            # 制度DSL 有効時のみ機械可読 rule を添える
                    npois = []
                    for ln in prompt.splitlines():
                        if ln.startswith("周りにある店・場所:"):
                            npois = [x for x in ln.split(":", 1)[1].strip().split("、")
                                     if x]
                    prop["rule"] = self._mock_rule(rng, place, npois)
                return json.dumps(prop, ensure_ascii=False)

        # 開放行動(第17バッチ・freedom ON のヘッダ行がある時だけ=OFF では draw も分岐も不変)
        if '"action": "do"' in prompt:
            t2 = rng.random()
            if t2 < 0.25:
                acts = ("公園を散歩する", "部屋で絵を描く", "本を読む",
                        "部屋の掃除をする", "音楽を聴いてのんびりする",
                        "近所の募金箱に少し寄付する")
                vals = (["感情"], ["認識"], ["実用"], ["社会"], [])
                return json.dumps(
                    {"action": "do",
                     "what": acts[int(rng.integers(len(acts)))],
                     "minutes": int(rng.integers(2, 13)) * 10,
                     "value": vals[int(rng.integers(len(vals)))]},
                    ensure_ascii=False)

        # 生活の自己決定 P2(D3・freedom.p2.* のメニュー "生活の選択" がある時だけ=OFFでは
        # draw も分岐も無い=バイト一致)。各行の JSON トークンが載っている項目だけ低確率で返す。
        if "生活の選択" in prompt:
            tp = rng.random()
            if '"action":"move_home"' in prompt and tp < 0.15:
                return json.dumps({"action": "move_home", "area": place},
                                  ensure_ascii=False)
            if '"permit":false' in prompt and 0.15 <= tp < 0.30:
                return json.dumps({"action": "open_venture", "name": f"{place}の露店",
                                   "offer": "軽食", "permit": False}, ensure_ascii=False)
            if '"action":"buy"' in prompt and 0.30 <= tp < 0.50:
                cat = ["food", "goods", "leisure"][int(rng.integers(3))]
                return json.dumps({"action": "buy", "cat": cat}, ensure_ascii=False)
            if '"action":"study"' in prompt and 0.50 <= tp < 0.65:
                return json.dumps({"action": "study", "topic": "歴史"},
                                  ensure_ascii=False)
            if '"action":"propose_partnership"' in prompt and 0.65 <= tp < 0.80:
                names = []
                for line in prompt.splitlines():
                    if line.startswith("近くにいる人:"):
                        names = [t for t in line.split(":", 1)[1].strip().split("、") if t]
                if names:
                    return json.dumps(
                        {"action": "propose_partnership",
                         "to": names[int(rng.integers(len(names)))]},
                        ensure_ascii=False)
            if '"action":"break_up"' in prompt and 0.80 <= tp < 0.90:
                return json.dumps({"action": "break_up"}, ensure_ascii=False)

        if "内省してください" in prompt:
            belief = f"{place}での経験から、自分は少し考えが変わった気がする。"
            out = {
                "action": "reflect", "belief": belief,
                "summary": "街を歩き、人と話し、いろいろ考えた一日だった。",
                "salient": [{"text": f"{place}での出来事が印象に残った",
                             "importance": int(rng.integers(3, 9))}],
            }
            if "自己像" in prompt:       # 深い内省(反射=自己モデル。第11バッチ。ON 時のみ到達)
                out["self"] = (f"{place}のあたりで過ごすうち、前より人と関わるのが"
                               "好きになってきた気がする。")
                out["ties"] = "よく顔を合わせる相手が、自分の居場所になりつつある。"
            return json.dumps(out, ensure_ascii=False)

        if "SNSに短く投稿" in prompt:
            text = f"{place}なう。今日はこんな感じ。"
            used = []
            if known_terms and rng.random() < 0.5:
                term = known_terms[int(rng.integers(len(known_terms)))]
                text = f"{place}なう。「{term}」ってやつだこれ。"
                used = [term]
            return json.dumps({"action": "post", "text": text, "use_terms": used},
                              ensure_ascii=False)
        if "メッセージを送る" in prompt:
            return json.dumps({"action": "dm",
                               "text": f"いま{place}にいるよ。あとで会える？"},
                              ensure_ascii=False)

        if surprised and roll < 0.25:
            # 状況に名前を付ける(造語)— ラベル/語彙の誕生
            word = self._coin_word(rng)
            text = f"この{place}の感じ、「{word}」って呼ぶことにするよ。"
            return json.dumps({"action": "coin_label", "word": word, "text": text},
                              ensure_ascii=False)
        if roll < 0.55:
            pool = _COMPLAINTS if surprised else _SMALLTALK
            text = pool[int(rng.integers(len(pool)))].format(place=place)
            used = []
            if known_terms and rng.random() < 0.5:
                term = known_terms[int(rng.integers(len(known_terms)))]
                text += f" まさに「{term}」だね。"
                used = [term]
            return json.dumps({"action": "speak", "text": text, "use_terms": used},
                              ensure_ascii=False)
        return json.dumps({"action": "wander"}, ensure_ascii=False)
