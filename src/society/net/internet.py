"""インターネット層(簡易版スマホ): SNS(X風)+ニュース+検索+DM+地図アプリ。

現実(対面)の層に重なる情報チャネル。伝播系譜(provenance)には
channel = "sns" / "news" / "search" / "dm" として記録され、
対面(face)との経路比較ができる(ユーザー指定の伝播経路ログの拡張)。

- SNS: 投稿(post)はタイムラインに載る。読む側はフォロー相手+全体の新着を見る。
- フォロー: 初期はランダム k 人 + 対面で会話した相手(contacts)は自動フォロー。
- ニュース: シナリオイベント(公式発表など)の配信面。author=-1(メディア)。
- 検索: 知らない言葉を聞いたら後でスマホで調べる(search_queue 経由)。
- DM: 対面で知り合った相手に 1対1 メッセージ(遠隔でも届く)。
状態はすべて agent_id 順の適用で更新される(D13 決定論)。
"""
from __future__ import annotations

MEDIA_ID = -1   # 公式・メディアの発信者 ID


class Internet:
    def __init__(self, feed_size: int = 6, posts_max: int = 0):
        self.feed_size = int(feed_size)
        # post: {"id","step","author","text","items":[word],"likes":set,"reshares":int}
        self.posts: list[dict] = []
        self.news: list[dict] = []     # {"step","title","text","items":[word]}
        self.follows: dict[int, set[int]] = {}
        self.contacts: dict[int, set[int]] = {}   # 対面で会話した相手(DM可)
        self.read_marks: dict[int, int] = {}      # agent_id -> 既読 post id(=watermark)
        # #13 タイムライン優先枠: reader_id -> {必ず載せる著者 id}(founder 投稿の到達保証)
        self.priority: dict[int, set[int]] = {}
        # B7(スケール): posts の有界化。post の id は追記単調増加の通し番号(= offset + 位置)
        # で、posts_max を超えたら先頭から破棄し offset を進める。id は不変なので react/
        # read_marks/rt_of は id ベースで安全。既定 0 では offset は常に 0 で従来と完全同一。
        self.posts_max = int(posts_max)
        self._post_offset = 0
        # B8(スケール): いいね/リシェアの増分カウンタ(毎step全走査を廃止)。offset で古い
        # post が消えても総数を保持する。既定(trim なし)では sum(len(likes)) と完全一致。
        self.n_likes_total = 0
        self.n_reshares_total = 0

    def init_follows(self, agent_ids: list[int], rng, k: int = 6) -> None:
        ids = list(agent_ids)
        for aid in ids:
            others = [x for x in ids if x != aid]
            n = min(k, len(others))
            picks = rng.choice(len(others), size=n, replace=False) if n else []
            self.follows[aid] = {others[int(i)] for i in picks}
            self.contacts[aid] = set()

    def add_contact(self, a: int, b: int) -> None:
        if a in self.contacts and b in self.contacts:
            self.contacts[a].add(b)
            self.contacts[b].add(a)
            self.follows[a].add(b)                # 知り合いは自動フォロー

    def follower_count(self, author: int) -> int:
        """author をフォローしている人数(= フォロワー数)。Wave G6 のインフルエンサー判定用。

        follows[x] は「x がフォローしている相手」なので逆引きで数える(決定論・乱数なし)。
        info_env.influence が ON のときだけ呼ばれる=既定挙動の draw/イベントには一切影響しない。"""
        a = int(author)
        return sum(1 for s in self.follows.values() if a in s)

    def set_priority(self, reader: int, author: int) -> None:
        """reader のタイムラインに author の投稿を必ず載せる(#13 founder 優先枠)。"""
        self.priority.setdefault(int(reader), set()).add(int(author))

    # ---- 発信 ----
    def post(self, author: int, text: str, items: list[str], step: int,
             event_id: int | None = None, rt_of: int | None = None) -> int:
        """投稿を追記し、その post_id(= 追記通し番号 = offset + 位置)を返す。"""
        pid = self._post_offset + len(self.posts)
        rec = {"id": pid, "step": step, "author": author, "text": text,
               "items": list(items), "likes": set(), "reshares": 0}
        if event_id is not None:                  # イベント参加告知(閲覧で invite に紐付く)
            rec["event_id"] = int(event_id)
        if rt_of is not None:                     # リシェア(元 post id)
            rec["rt_of"] = int(rt_of)
        self.posts.append(rec)
        if self.posts_max > 0 and len(self.posts) > self.posts_max:
            drop = len(self.posts) - self.posts_max
            del self.posts[:drop]                 # 古い post をエイジアウト
            self._post_offset += drop             # id は不変・offset で位置を補正
        return pid

    # ---- 反応(#14 いいね・リシェア)。状態更新のみ(判定はしない)----
    def react(self, reader: int, post_id: int, kind: str, step: int,
              author_name: str | None = None) -> int | None:
        """post へのいいね/リシェア。元著者 id を返す(範囲外なら None)。

        - like: likes に reader を追加。
        - reshare: reshares を +1 し、"RT @元著者名: 本文" を新規 post として追記
          (items は元と同一、rt_of=元 post id)。フォロワーへは既存 timeline 配信で
          自然に再配信され、カスケードは _hear_words("sns") に自動で乗る。
        """
        idx = int(post_id) - self._post_offset      # id → 現在の位置(古い post は消えている)
        if idx < 0 or idx >= len(self.posts):
            return None
        post = self.posts[idx]
        author = post["author"]
        if kind == "like":
            likes = post["likes"]
            if int(reader) not in likes:            # set 意味を保ちつつ増分カウント(B8)
                likes.add(int(reader))
                self.n_likes_total += 1
        elif kind == "reshare":
            post["reshares"] += 1
            self.n_reshares_total += 1              # 増分カウント(B8)
            name = author_name if author_name is not None else str(author)
            self.post(int(reader), f"RT @{name}: {post['text']}",
                      list(post["items"]), step, rt_of=post_id)
        return author

    def amplify_reshare(self, post_id: int, extra: int) -> None:
        """リシェア到達の加重(Wave G6 バイラル)。post の reshares カウンタを extra だけ増やす。

        インフルエンサー(高フォロワー)の投稿が「遠くまで届いた」ことを非LLM の状態更新=拡散
        カウンタで表現する(新しい投稿=heard 語の流入を作らない=物理不変の観測量に留める)。
        新規 generate を1本も足さず、既存 post を1件も追加しない(reach は数として記録)。"""
        idx = int(post_id) - self._post_offset
        if idx < 0 or idx >= len(self.posts):
            return
        self.posts[idx]["reshares"] += int(extra)
        self.n_reshares_total += int(extra)

    def publish_news(self, title: str, text: str, items: list[str],
                     step: int) -> None:
        self.news.append({"step": step, "title": title, "text": text,
                          "items": list(items)})
        # 公式発表は SNS にも流れる(現実のプレスリリース挙動)
        self.post(MEDIA_ID, f"【{title}】{text}", items, step)

    # ---- 閲覧 ----
    def timeline_for(self, aid: int) -> list[dict]:
        """未読の新着から、フォロー相手優先で feed_size 件。

        priority(#13)を持つ読者は、その著者の新着を必ず含め、残枠を従来ロジックで
        埋める。priority が空の読者は**完全に旧ロジック**(バイト一致)。
        """
        start = self.read_marks.get(aid, 0)          # 既読 watermark(post id)
        start_idx = start - self._post_offset        # id → 位置。古い post は消えている
        if start_idx < 0:
            start_idx = 0                            # 未読の古い post が退避済みなら先頭から
        fresh = [p for p in self.posts[start_idx:] if p["author"] != aid]
        self.read_marks[aid] = self._post_offset + len(self.posts)
        follows = self.follows.get(aid, set())
        followed = [p for p in fresh if p["author"] in follows
                    or p["author"] == MEDIA_ID]
        rest = [p for p in fresh if p not in followed]
        prio = self.priority.get(aid)
        if not prio:
            return (followed + rest)[-self.feed_size:]   # 旧ロジック完全一致
        # 優先著者の新着を必ず含める(多すぎれば直近 feed_size 件)。
        prio_posts = [p for p in fresh if p["author"] in prio][-self.feed_size:]
        remaining = self.feed_size - len(prio_posts)
        conv = [p for p in (followed + rest) if p not in prio_posts]
        conv_keep = conv[-remaining:] if remaining > 0 else []
        return conv_keep + prio_posts

    def ranked_timeline_for(self, aid: int, score_of):
        """推薦=エコーチェンバー(Wave G6): fresh 候補を意見整合 score で選別する。

        通常の timeline_for が「時系列(recency)」で feed_size 件を選ぶのに対し、こちらは
        score_of(post)(高いほど閲覧者の意見に整合)の上位 feed_size 件を選ぶ=整合投稿を
        優先表示し、不整合投稿を間引く(分極の加速器)。既読 watermark の進め方は timeline_for
        と完全に同一(既読 = posts 末尾)。返り値は (feed(recency 昇順表示), boosted, filtered):
          boosted  = 整合ゆえ従来窓に無かったのに引き上げられた post id
          filtered = 不整合ゆえ従来なら見えた窓から押し出された post id
        候補が feed_size 以下なら選別不要=従来と同じ集合(boosted/filtered は空)。決定論・乱数なし。
        """
        start = self.read_marks.get(aid, 0)
        start_idx = start - self._post_offset
        if start_idx < 0:
            start_idx = 0
        fresh = [p for p in self.posts[start_idx:] if p["author"] != aid]
        self.read_marks[aid] = self._post_offset + len(self.posts)
        if len(fresh) <= self.feed_size:
            return list(fresh)[-self.feed_size:], [], []
        plain_ids = {p["id"] for p in fresh[-self.feed_size:]}   # 従来(時系列)窓
        ranked = sorted(fresh, key=lambda p: (score_of(p), p["id"]))  # 昇順(末尾=最整合)
        selected = ranked[-self.feed_size:]
        sel_ids = {p["id"] for p in selected}
        boosted = sorted(sel_ids - plain_ids)
        filtered = sorted(plain_ids - sel_ids)
        feed = sorted(selected, key=lambda p: p["id"])           # 表示は recency(id 昇順)
        return feed, boosted, filtered

    def ranked_exposure_timeline_for(self, aid: int, weight_of):
        """露出重み付き TL(社会的ヒエラルキー ON): fresh 候補を weight_of(post) 上位で選ぶ。

        優先的選択(Barabási)/威信の可視化: weight_of(post)(高いほど地位・被フォロー数が大きい
        投稿者)の上位 feed_size 件を選ぶ=高地位・ハブ投稿の露出を厚くする。既読 watermark の
        進め方は timeline_for / ranked_timeline_for と完全に同一(既読=posts 末尾)。返り値の件数は
        timeline_for と同じ min(len(fresh), feed_size) 件=react の draw 数は不変。候補が feed_size 以下
        なら選別不要(従来と同じ集合)。表示は recency(id 昇順)。決定論・乱数なし。"""
        start = self.read_marks.get(aid, 0)
        start_idx = start - self._post_offset
        if start_idx < 0:
            start_idx = 0
        fresh = [p for p in self.posts[start_idx:] if p["author"] != aid]
        self.read_marks[aid] = self._post_offset + len(self.posts)
        if len(fresh) <= self.feed_size:
            return fresh[-self.feed_size:]
        ranked = sorted(fresh, key=lambda p: (weight_of(p), p["id"]))  # 昇順(末尾=最重)
        selected = ranked[-self.feed_size:]
        return sorted(selected, key=lambda p: p["id"])               # 表示は recency(id 昇順)

    def latest_news(self, n: int = 3) -> list[dict]:
        return self.news[-n:]
