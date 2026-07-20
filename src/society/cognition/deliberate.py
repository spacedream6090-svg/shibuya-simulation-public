"""deliberate 層(LLM 経路)。プロンプト構成→生成→JSON 解釈。失敗は routine へ(D16)。

プロンプトは APC 効率のため「共通部を先頭、個別部を後ろ」に置く(infra 検証の帰結)。
"""
from __future__ import annotations

import json

from ..factors.mood import mood_text

# ヘッダは run 内固定(labeling_mode は run を通して不変)なので APC の prefix 一致は保たれる。
# constrained(既定)の文言は現行と一字一句同一に保つ(既定の再現性=キャッシュ整合のため)。
# 街の名前(冒頭の {city})は基盤に持たせず cfg(envpack.lexicon.place_name)から受ける。
# 既定値は config.yaml の envpack が供給するので出力はバイト一致(ゴールデン維持)。
_HEADER_INTRO = (
    "あなたは{city}の街で暮らす一人の人間です。状況に対して自然に振る舞ってください。\n"
)
_HEADER_FORMS = (
    "出力は次のいずれかの JSON 1個のみ(キー名は厳守):\n"
    '  {"action": "speak", "text": "話す内容"}\n'
)
_COIN_LINE = {
    "constrained": '  {"action": "coin_label", "word": "新しい呼び名", "text": "それを使った一言"}\n',
    "open": '  {"action": "coin_label", "word": "新しい言い回し・フレーズでもよい", "text": "それを使った一言"}\n',
}
_HEADER_TAIL = (
    '  {"action": "post", "text": "SNSに投稿する内容"}\n'
    '  {"action": "dm", "text": "メッセージの内容"}\n'
    '  {"action": "wander"}\n'
)
# 開放行動(第17バッチ・freedom.open_actions=true のときのみ足す1行。既定 OFF=ヘッダ不変)。
# 中立提示: 例を挙げない・促さない(coin_label と同じ流儀)。value は任意の自己申告
# (観測のみ。書かなくてもよい)。
_DO_LINE = (
    '  {"action": "do", "what": "他にしたいこと(自由に書く)", '
    '"where": "する場所の名前(任意)", "minutes": 30, '
    '"value": ["実用/感情/社会/認識 のうち自分が感じるもの(任意)"]}\n'
)


def _header(labeling_mode: str, open_actions: bool = False,
            city_name: str = "") -> str:
    coin = _COIN_LINE.get(labeling_mode, _COIN_LINE["constrained"])
    do = _DO_LINE if open_actions else ""
    head = _HEADER_INTRO.format(city=city_name) + _HEADER_FORMS
    return head + coin + _HEADER_TAIL + do


# ヘッダの構造ベースライン(街名は呼び出し時に注入=ここでは空)。後方互換の基準。
_COMMON_HEADER = _header("constrained")

# 「所持ツール」節(標準装備の中立告知)。tools.equip_all=true のとき全発火プロンプトに
# 固定順・固定書式で注入する。中立記述に徹する(勧誘語「〜しましょう/おすすめ/ぜひ」禁止):
# 行動名 + 中立な機能説明 + JSON 形式 + 客観条件のみ。因子名や「世界改変」系の語は書かない
# (指紋回避 = tests/test_contracts の禁止語に触れない)。mock の分岐マーカー("他にできる
# こと"/"驚き:"/"場所:" 等)とは別文字列 "所持ツール" なので mock の分岐を誤爆させない。
# 順は propose/host_event/post_flyer/found_group/open_venture(固定)。条件は客観
# (場所・所持金)のみ = k 非依存(R1)。
_EQUIP_INTRO = "所持ツール(いつでも使える。使っても使わなくてもよい):"
_EQUIP_LINES = (
    '- propose(提案): 街の取り決めを提案する。賛同が集まると成立する。'
    '形式 {"action":"propose","text":"…"}(条件: なし)',
    '- host_event(イベント開催): 勉強会や集まりを企画する。参加者が集まる。'
    '形式 {"action":"host_event","title":"…","hours_later":2}(条件: なし)',
    '- post_flyer(ビラ掲示): 今いる場所に貼り紙を出す。通りかかった人が見る。'
    '形式 {"action":"post_flyer","text":"…"}(条件: なし)',
    '- found_group(コミュニティ結成): グループを作る。'
    '名前を聞いた知人が加わることがある。'
    '形式 {"action":"found_group","name":"…","purpose":"…"}(条件: なし)',
)


def _equip_section(agent, venture_cost: float) -> str:
    """標準装備ツール節を組み立てる(決定論・k 非依存)。open_venture の可否は所持金の
    客観条件のみで判定する(R1: k とは無関係)。money は観測可能な客観量で因子ではない。"""
    cost = int(venture_cost)
    money = int(getattr(agent, "money", 0) or 0)
    status = f"利用可(所持金{money}円)" if agent.money >= venture_cost \
        else f"不足(所持金{money}円)"
    venture = ('- open_venture(出店): 屋台・店を開く。通行人が買う。'
               '形式 {"action":"open_venture","name":"…","offer":"…"}'
               f'(条件: 所持金{cost}円以上。現在: {status})')
    return "\n".join((_EQUIP_INTRO, *_EQUIP_LINES, venture))


_ACTIVITY_JP = {"working": "仕事中", "commuting": "通勤の途中",
                "eating": "食事中", "shopping": "買い物中"}


def _time_label(sim_min: int) -> str:
    m = sim_min % 1440
    h, mm = divmod(m, 60)
    part = ("深夜" if h < 5 else "朝" if h < 10 else "昼" if h < 16
            else "夕方" if h < 19 else "夜")
    return f"{h:02d}:{mm:02d}({part})"


def build_prompt(agent, *, place_name: str, surprise: str | None,
                 nearby_names: list[str], sim_min: int = 0, step: int = 0,
                 nearby_pois: list[str] | None = None,
                 nearby_ids: list[int] | None = None,
                 dm_target: str | None = None,
                 feed_texts: list[str] | None = None,
                 reply_to: tuple[str, str] | None = None,
                 tool_offers: str | None = None,
                 memberships: list[str] | None = None,
                 pull_query: str | None = None,
                 familiar_places: list[str] | None = None,
                 institutions: list[str] | None = None,
                 equip_all: bool = False,
                 venture_cost: float = 30000.0,
                 date_line: str | None = None,
                 weather_line: str | None = None,
                 schedule_line: str | None = None,
                 event_line: str | None = None,
                 disaster_line: str | None = None,
                 ads_line: str | None = None,
                 crowd_line: str | None = None,
                 wv_expect_line: str | None = None,
                 wv_self_line: str | None = None,
                 wv_norm_line: str | None = None,
                 relation_line: str | None = None,
                 reputation_line: str | None = None,
                 household_line: str | None = None,
                 diversity_line: str | None = None,
                 emotion_line: str | None = None,
                 goal_line: str | None = None,
                 hobby_line: str | None = None,
                 norm_line: str | None = None,
                 digest_line: str | None = None,
                 interstitial_digest: str | None = None,
                 scene_lines: list[str] | None = None,
                 variety_hint: bool = False,
                 labeling_mode: str = "constrained",
                 open_actions: bool = False,
                 city_name: str = "",
                 p2_offers: str | None = None,
                 dialog_history: list | None = None) -> str:
    """個別文脈(時刻・場所・活動・気分・記憶・直近発話)を渡し、内容の固定化を防ぐ。

    pull_query が渡された時だけ(agentic_pull=true)、その文で決定論の記憶想起を
    1回行い1行注入する。LLM 呼び出しは増やさない(発火時の pull はここに集約)。
    labeling_mode(constrained 既定 / open)はヘッダの coin_label 行のみを差し替える。
    open_actions(第17バッチ・既定 False)は開放行動 "do" の1行だけをヘッダ末尾に足す。"""
    # 入力解像度LOD(第30バッチ・lod.input_res)。OFF=属性なし → 既定値=現行定数で
    # バイト一致。ON でも変わるのは注入の「件数」だけ(呼数・乱数・発火は不変=R1)。
    # beliefs(k の行動流入路 D7)と全員共通行は解像度の対象外(docs/plans/input-resolution-lod.md §2)。
    _ir = getattr(agent, "input_res", None) or {}
    _poi_n = int(_ir.get("poi_n", 3))
    _people_n = int(_ir.get("people_n", 0))        # 0 = 全列挙(現行)
    _recent_n = int(_ir.get("recent_n", 4))
    _retrieve_n = int(_ir.get("retrieve_n", 3))
    _feed_n = int(_ir.get("feed_n", 3))
    lines = [_header(labeling_mode, open_actions, city_name), agent.persona]
    # 反射=自己モデル(第11バッチ 2026-07-08。深い内省の産物。OFF は None=行なし=バイト一致)。
    # persona(固定の自己紹介)の直後に「経験から更新される自己理解」を置く=自己認識の再帰。
    sm = getattr(agent, "self_model", None)
    if sm and sm.get("self"):
        sm_line = f"自分の理解(内省より): {sm['self']}"
        if sm.get("ties"):
            sm_line += f" / 大事な関係: {sm['ties']}"
        lines.append(sm_line)
    # 無意識層(第12バッチ。行動からの自己知覚=揮発的な作動自己。OFF は空文字=行なし=不変)。
    imp = getattr(agent, "implicit_self", "")
    if imp:
        lines.append(f"最近の自分(なんとなく感じていること): {imp}")
    membership_line = getattr(agent, "org_line", None)
    if membership_line:                  # 所属1行(organizations 有効時のみ設定。無ければ不変)
        lines.append(membership_line)
    if date_line:                        # 日付・暦(world.calendar 有効時のみ。全員共通・k非依存)
        lines.append(date_line)
    if weather_line:                     # 天気(weather 有効時のみ。全員共通・k非依存)
        lines.append(weather_line)
    if schedule_line:                    # 近い予定(schedule 有効時のみ。会話由来・k非依存)
        lines.append(schedule_line)
    if event_line:                       # 年中行事(annual_events 有効かつ当日のみ。全員共通・k非依存)
        lines.append(event_line)
    if disaster_line:                    # 都市・環境ショック(disaster 有効かつ発生中のみ。全員共通・k非依存)
        lines.append(disaster_line)
    if ads_line:                         # 街頭広告の想起(ads 有効かつ想起窓内のみ。中立提示・第18バッチ)
        lines.append(ads_line)
    lines += [f"時刻: {_time_label(sim_min)}",
              f"場所: {place_name}"]
    if nearby_pois:
        lines.append(f"周りにある店・場所: {'、'.join(nearby_pois[:_poi_n])}")
    if scene_lines:                      # 構造化シーン記述 v0(scene_desc 有効時のみ。方向つき視界/
        lines.extend(scene_lines)        # 注視対象/垂直関係。既定 OFF は None=1行も足さない=バイト一致)
    if wv_expect_line:                   # 場所の期待vs実際(worldview 有効かつ差が大きい時のみ。第20バッチ)
        lines.append(wv_expect_line)
    if institutions:                     # Searle 制度化(ON時のみ。全員平等・k非依存の1行)
        lines.append(f"この街の取り決め: {' / '.join(institutions[:2])}")
    if norm_line:                        # 再帰性: いま実効の取り決め(recursion 有効時のみ。全員共通・k非依存)
        lines.append(norm_line)
    if digest_line:                      # 再帰性: 昨日の街の動き(客観カウント。全員共通・k非依存)
        lines.append(digest_line)
    if interstitial_digest:              # 行間補間(P2 S2): 前回発火以降の客観ダイジェスト。
        lines.append(interstitial_digest)  # 既定 None=1行も足さない=バイト一致(digest_line と同型)
    if wv_norm_line:                     # 記述規範: 新しいことを始める人がいる街か(worldview 有効時のみ。全員共通)
        lines.append(wv_norm_line)
    act = _ACTIVITY_JP.get(agent.activity)
    if act:
        lines.append(f"いま: {act}")
    lines.append(f"気分: {mood_text(agent.states)}")
    if wv_self_line:                     # 世界への手応え(worldview 有効かつ閾値超えのみ。自然文=因子語なし)
        lines.append(wv_self_line)
    if emotion_line:                     # 離散感情ラベル(内面 H6・affect ON 前提。既定 OFF は None=不変)
        lines.append(emotion_line)
    if agent.beliefs:                    # 内省の書き戻し先 = k の行動への流入経路(D7)
        lines.append(f"あなたの考え(これまでの内省から): "
                     f"{' / '.join(agent.beliefs[-3:])}")
    if nearby_names:
        # 注意の幅(people_n>0 のときだけ列挙を絞る。間柄 relation_line は nearby_ids
        # ベースで不変=「見えているが名前として意識に上らない」の近似)
        _names = nearby_names[:_people_n] if _people_n > 0 else nearby_names
        lines.append(f"近くにいる人: {'、'.join(_names)}")
    if crowd_line:                       # 群衆の視覚情報(crowd_visual 有効時のみ。実在集計=第18バッチ)
        lines.append(crowd_line)
    if agent.adopted:
        lines.append(f"知っている言葉: {'、'.join(sorted(agent.adopted))}")
    if familiar_places:                  # Lynch 認知地図(ON時のみ。よく知っている場所 上位3)
        lines.append(f"馴染みの場所: {'、'.join(familiar_places[:3])}")
    recent = agent.mem.recent(_recent_n)
    if recent:
        lines.append(f"直近の出来事: {' / '.join(recent)}")
    recalled = agent.mem.retrieve(step, [place_name] + list(nearby_names or []),
                                  n=_retrieve_n, agent_id=agent.id)
    if recalled:
        lines.append(f"記憶に残っていること: {' / '.join(recalled)}")
    if pull_query:                       # agentic pull(発火時。決定論・呼び出しは増やさない)
        # ACT-R OFF(actr is None)は failed 常に False=hits は従来 query と同一=バイト一致。
        # ON のとき、手掛かりはあるが全候補が閾値未達なら「思い出そうとして失敗」を1行にする
        # (ここは build_prompt=logger 不在の非意図的な "ふと思い出す" 経路なので memory_fail
        #  イベントは出さず1行のみ。意図的な内省 pull は reflection 側でイベントも発火する)。
        res = agent.mem.query_ex(step, pull_query, n=2, agent_id=agent.id)
        if res.hits:
            lines.append(f"ふと思い出したこと: {' / '.join(res.hits)}")
        elif res.failed:
            lines.append(f"({res.cue}のことを思い出そうとしたが、はっきりしない…)")
    # 間柄: relations 有効時は tier/派閥つきの拡張行(scheduler が事前構築)。既定 OFF は
    # relation_line=None で従来の count ベース relation_line に後退=バイト一致。
    rel = relation_line if relation_line is not None \
        else agent.mem.relation_line(nearby_ids or [])
    if rel:
        lines.append(f"間柄: {rel}")
    if reputation_line:                  # 評判(relations 有効かつ閾値超のときのみ。既定 OFF は None)
        lines.append(reputation_line)
    if household_line:                   # 世帯・恋愛(household 有効かつ同席者に同居者/恋人。既定 OFF は None)
        lines.append(f"同席の身近な人: {household_line}")
    if diversity_line:                   # 観光・多言語(diversity 有効かつ観光客/非日本語話者のみ。既定 OFF は None)
        lines.append(diversity_line)
    if goal_line:                        # 長期目標(内面 H6・keystone 駆動源。既定 OFF は None=不変)
        lines.append(goal_line)
    if hobby_line:                       # 趣味・関心(内面 H6・サブカルチャー文脈。既定 OFF は None=不変)
        lines.append(hobby_line)
    if agent.mem.day_summaries:
        lines.append(f"昨日までの日記: {agent.mem.day_summaries[-1]}")
    if agent.said:
        lines.append(f"あなたがさっき言ったこと: {' / '.join(agent.said[-2:])}")
        lines.append("注意: さっきと同じ話題・言い回しを繰り返さない。"
                     "今の時刻・場所・気分・出来事に根ざした新しい内容を話す。")
    if feed_texts:
        lines.append(f"タイムライン: {' / '.join(feed_texts[:_feed_n])}")
    # 対話履歴の注入(会話強化・prompts.dialog_history=true のときだけ scheduler が渡す)。
    # 相手との直近やりとり(最大2往復=4発話)を1行で。既定 OFF は None=1行も足さない
    # =バイト一致(ゴールデン維持)。行数は最大1行増(2往復を「→」で連結)。
    if dialog_history:
        turns = dialog_history[-4:]          # 直近2往復=最大4発話
        lines.append("直前のやりとり: "
                     + " → ".join(f"{spk}「{txt}」" for spk, txt in turns))
    if surprise == "social":
        lines.append("状況: 近くにいる人と自然に会話する。今この場所・この時間ならではの"
                     "話題(店、時間帯、出来事、気分)で、あなたらしい一言を speak で。")
        if variety_hint:                 # 改善 P3(既定 OFF): 定型の情景報告で始めない
            lines.append(f"書き出しの注意: 「この時間の{city_name}は…」のような情景報告の"
                         "決まり文句で始めない。具体的な出来事・固有名詞・相手への"
                         "問いかけなど、毎回違う切り口で。")
    elif surprise == "reply" and reply_to is not None:
        lines.append(f"状況: {reply_to[0]}に話しかけられた:「{reply_to[1]}」。"
                     "相手に自然に返事をする(speak で)。")
        if variety_hint:                 # 改善 P3(既定 OFF): 返答もオウム返し・定型を避ける
            lines.append("返事の注意: 相手の言葉のオウム返しや決まり文句を避け、"
                         "自分の経験・予定・意見を一つ足して返す。")
    elif surprise == "post":
        lines.append("状況: スマホでSNSを開いた。いま感じていること、目にした光景、"
                     "タイムラインへの反応など、何か一つを短くつぶやく(post で)。"
                     "SNSらしいくだけた口語の短文。場所や時刻の報告文にしない。"
                     "挨拶・自己紹介もしない。")
    elif surprise == "dm":
        lines.append(f"状況: {dm_target}にスマホでメッセージを送る(dm で)。"
                     "近況や誘い、さっき見聞きしたことなど自然な内容で。")
    elif surprise == "solo":
        lines.append("状況: ふと立ち止まって考え事。今の場所・時間帯・気分・最近の"
                     "出来事から思ったことを一言(speak。独り言でよい)。")
        if variety_hint:                 # 改善 P3(既定 OFF): 独り言も情景報告の定型を避ける
            lines.append("注意: 場所と時刻をなぞる報告文にせず、気になっていること・"
                         "思い出したこと・これからの予定など内面から出る一言に。")
    elif surprise:
        jp = {"congestion": "ひどい混雑に巻き込まれた",
              "novel_place": "初めて来る場所だ",
              "unknown_word": "知らない言葉を耳にした"}[surprise]
        lines.append(f"驚き: {jp}")
    if memberships:                      # 所属の自覚(グループのメンバーであること)
        lines.append(f"あなたは次のグループのメンバー: {'、'.join(memberships)}")
    if tool_offers:                      # 中立提示(選択肢。使用は勧めない=造語と同じ扱い)
        lines.append(tool_offers)
    if equip_all:                        # 標準装備の中立告知(勧誘なし。R1: 客観条件のみ)
        lines.append(_equip_section(agent, venture_cost))
    if p2_offers:                        # 生活の自己決定 P2(freedom.p2.* 有効時のみ。中立提示・客観条件つき)
        lines.append(p2_offers)          # 促進・誘導なし(scheduler が客観条件で組み立て済み)
    return "\n".join(lines)


# ---- 行動方針キャッシュ P2 S7: 日中熟慮の cognition 層の関数入口 ----------------- #
# scheduler の熟慮ディスパッチ(build_prompt→generate→parse_action)へ主計画者が差し込む
# 薄い seam。既定 OFF は policy_cache 側が完全 no-op(cache も stream も作らず=バイト一致)。
#   使い方(主計画者の統合時):
#     action = maybe_reuse_action(sim, agent, step, sim_min, trigger)
#     if action is None:
#         ... build_prompt → generate → parse_action で action を得る ...
#         store_action(sim, agent, step, sim_min, trigger, action)
def maybe_reuse_action(sim, agent, step: int, sim_min: int, trigger: str):
    """熟慮の再利用を試みる。命中+ゲート通過なら action(dict)を返し LLM をスキップ、
    そうでなければ None(=通常どおり LLM 生成)。既定 OFF は即 None=バイト一致。"""
    from . import policy_cache as _policy_cache      # 遅延 import(循環回避)
    return _policy_cache.reuse_action(sim, agent, step, sim_min, trigger)


def store_action(sim, agent, step: int, sim_min: int, trigger: str, action) -> None:
    """LLM 生成した熟慮 action をキャッシュへ格納(既定 OFF は no-op)。"""
    from . import policy_cache as _policy_cache      # 遅延 import(循環回避)
    _policy_cache.store_action(sim, agent, step, sim_min, trigger, action)


def _loads_lenient(response: str) -> dict | None:
    """途中で切れた JSON を軽く修復して読む(トークン上限での閉じ括弧欠落など)。"""
    if not isinstance(response, str):
        return None
    stripped = response.strip()
    for candidate in (response, stripped, stripped + "}", stripped + '"}',
                      stripped + '"}' + "}"):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def parse_action(response: str) -> dict | None:
    """JSON を解釈。壊れていたら None(→呼び出し側が routine へフォールバック)。"""
    data = _loads_lenient(response)
    if data is None:
        return None
    if not isinstance(data, dict) or "action" not in data:
        return None
    kind = data["action"]

    def _text_of(*keys: str) -> str | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    if kind == "speak":
        text = _text_of("text", "content", "message", "say", "speech")
        if text:
            return {"type": "speak", "text": text,
                    "use_items": [t for t in data.get("use_terms", [])
                                  if isinstance(t, str)]}
        return None
    if kind == "coin_label":
        word = _text_of("word", "label", "name")
        if word:
            return {"type": "coin_label", "word": word,
                    "text": _text_of("text", "content", "message") or word}
        return None
    if kind == "post":
        text = _text_of("text", "content", "message")
        if text:
            return {"type": "post", "text": text,
                    "use_items": [t for t in data.get("use_terms", [])
                                  if isinstance(t, str)]}
        return None
    if kind == "dm":
        text = _text_of("text", "content", "message")
        if text:
            return {"type": "dm", "text": text}
        return None
    if kind == "host_event":
        title = _text_of("title", "name", "text")
        if title:
            try:
                hours = int(data.get("hours_later", 1))
            except (TypeError, ValueError):
                hours = 1
            return {"type": "host_event", "title": title,
                    "hours_later": max(1, min(6, hours))}
        return None
    if kind == "post_flyer":
        text = _text_of("text", "content", "message")
        if text:
            return {"type": "post_flyer", "text": text}
        return None
    if kind == "found_group":
        name = _text_of("name", "group", "title")
        if name:
            return {"type": "found_group", "name": name,
                    "purpose": _text_of("purpose", "text", "goal") or ""}
        return None
    if kind == "propose":
        text = _text_of("text", "proposal", "content", "message")
        if text:
            out = {"type": "propose", "text": text}
            rule = data.get("rule")          # 制度DSL: 機械可読の制度スロット(任意)
            if isinstance(rule, dict):       # 深い検証は rules.RuleBook が成立時に行う
                out["rule"] = rule
            return out
        return None
    if kind == "open_venture":
        name = _text_of("name", "title")
        if name:
            out = {"type": "open_venture", "name": name,
                   "offer": _text_of("offer", "text", "content") or ""}
            # 逸脱(#10 deviance): 無許可出店の申告。permit が偽(bool/文字列)なら通す
            # (裁定側=tools が deviance 有効時のみ許可待ちをスキップし permitted:false で開店)。
            permit = data.get("permit")
            if permit is False or (isinstance(permit, str)
                                   and permit.strip().lower() in ("false", "no", "なし")):
                out["permit"] = False
            return out
        return None
    # ---- 生活の自己決定 P2(#6-#10。OFF 時は提示されないだけで解釈は常に受ける=寛容) ----
    if kind == "move_home":              # #6 住居移転
        return {"type": "move_home", "area": _text_of("area", "region", "where", "place")}
    if kind == "buy":                    # #7 消費の意思(cat は裁定側で正準化)
        return {"type": "buy", "cat": _text_of("cat", "category", "what", "item")}
    if kind == "study":                  # #8 学び直し(効果は記録のみ)
        return {"type": "study", "topic": _text_of("topic", "subject", "what", "text")}
    if kind == "propose_partnership":    # #9 交際の申込
        to = _text_of("to", "name", "target", "partner")
        if to:
            return {"type": "propose_partnership", "to": to}
        return None
    if kind == "break_up":               # #9 別れ
        return {"type": "break_up"}
    if kind in ("do", "free", "activity"):   # 開放行動(第17バッチ。OFF 時は提示されないだけで
        what = _text_of("what", "text", "content", "action_desc")  # 解釈は常に受ける=寛容)
        if what:
            try:
                minutes = int(data.get("minutes", 30))
            except (TypeError, ValueError):
                minutes = 30
            return {"type": "free_action", "what": what[:120],
                    "where": _text_of("where", "place", "at"),
                    "minutes": max(10, min(240, minutes)),
                    "value_report": data.get("value")}
        return None
    if kind == "plan":                   # 朝の一日計画(cognition/planning.py が消費)
        items = [it for it in (data.get("items") or []) if isinstance(it, dict)]
        if items:
            return {"type": "plan", "items": items}
        return None
    if kind == "wander":
        return {"type": "stay"}
    if kind == "recall":                 # 内省の agentic pull 第1段(何を思い出すか)
        return {"type": "recall",
                "query": _text_of("query", "text", "content") or ""}
    if kind == "reflect":
        belief = _text_of("belief", "conclusion", "text", "content")
        summary = data.get("summary")
        salient = []
        for s in (data.get("salient") or []):
            if isinstance(s, dict) and isinstance(s.get("text"), str) and s["text"]:
                try:
                    imp = float(s.get("importance", 5))
                except (TypeError, ValueError):
                    imp = 5.0
                salient.append((s["text"].strip(), imp))
        if belief or summary or salient:
            out = {"type": "reflect", "belief": belief,
                   "summary": summary.strip() if isinstance(summary, str) else None,
                   "salient": salient}
            # 反射=自己モデル(第11バッチ): 深い内省の self/ties を通す。
            # 通常の内省 JSON には無いキー=無ければ省略(従来と完全同一)。
            self_txt = _text_of("self", "self_image")
            ties_txt = _text_of("ties", "relationships")
            if self_txt:
                out["self"] = self_txt
            if ties_txt:
                out["ties"] = ties_txt
            return out
    return None
