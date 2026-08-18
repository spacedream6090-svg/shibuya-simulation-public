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
        # ★被フォロー数の逆索引(author -> フォロワー数)。docs/research/
        #   initial-relations-improvement.md §1.2 が見つけた性能バグの根治:
        #   `follower_count` は `follows` の**全走査**だったので、`hierarchy.enabled`
        #   の日境界 `status._recompute` が全員ぶん呼ぶと O(N²) になり、25 万人で
        #   **1 日あたり約 36 分**を焼いていた(退場しても follows を刈らないので日を追って悪化)。
        #   ここを O(1) にする。**挙動は完全に不変**(数える対象も値も同一)。
        #   ★不変条件: `follows` を書き換える経路は `init_follows` / `ensure` /
        #     `add_contact` / `follow` / `set_follows` / `unfollow` の 6 つだけで、
        #     そのすべてが本索引を同時に更新する。`net.follows[x].add(y)` のように
        #     **直接**触ると索引が古くなるので、外部からは必ず `follow()` /
        #     `unfollow()` を使うこと
        #     (同値性は tests/test_follower_index.py が旧走査と突き合わせて固定する)。
        #   ★`add_contact(..., auto_follow=False)` は上の 6 経路のうち **follows を
        #     書かない**呼び方である(第117 SNC v2 の C1/C3。契約は「触らない」なので
        #     索引も動かない = 不変条件は保たれる)。既定 True は現行どおり書く。
        #   ★`unfollow` は SNC v2(net.contact_formation)の**上限の押し出し専用**として
        #     第117 で足した 6 本目の経路である(自然なアンフォロー = 関係の消滅は
        #     本選後の別件。設計 docs/plans/sns-contact-redesign.md §2)。
        self._followers: dict[int, int] = {}
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
        """起動時 1 回: 全員へ「自分以外から重複なし k 人」の初期フォローを配る。

        ★O(N²) だった理由と直し方(レーンP C1): 旧実装は 1 人ぶんごとに
          ``others = [x for x in ids if x != aid]``(= N-1 件の実体リスト)を作り直していたので、
          25 万人では 625 億回の比較 = **実測 0.011 秒 × 25 万人 ≒ 45 分**を起動時に焼いていた。
          抽選 ``rng.choice(n_pop, size=n, replace=False)`` は numpy Generator の Floyd 法で、
          **母集団の大きさ n_pop と本数 n だけ**から乱数を引く(``others`` の中身は見ない)。
          よって「添字 → 実 id」の写像を『自分の位置以上なら添字+1』で与えれば、
          **引かれる乱数列も選ばれる集合も 1 ビットも変えずに** O(N·k) になる。
          (同値性・乱数消費列の一致は tests/test_internet_entry_scale.py が旧実装と突き合わせる。)
        ★``ids`` に重複があるときだけ旧経路へ退避する(``others`` の長さが N-1 にならないので
          写像が成り立たない)。実運用の agent id は一意なのでこの枝は通らない。
        """
        ids = list(agent_ids)
        pos_of: dict[int, int] = {}
        for i, x in enumerate(ids):
            if x in pos_of:
                pos_of = {}                # 重複あり = 写像が使えない → 旧経路(下の分岐)
                break
            pos_of[x] = i
        unique = bool(pos_of) or not ids
        for aid in ids:
            if not unique:                 # 退避経路(重複 id): 旧実装そのまま
                others = [x for x in ids if x != aid]
                n = min(k, len(others))
                picks = rng.choice(len(others), size=n, replace=False) if n else []
                self.follows[aid] = {others[int(i)] for i in picks}
            else:
                pos = pos_of.get(aid)      # None = aid が名簿に無い(others = ids そのもの)
                n_pop = len(ids) - 1 if pos is not None else len(ids)
                n = min(k, n_pop)
                picks = rng.choice(n_pop, size=n, replace=False) if n else []
                if pos is None:
                    self.follows[aid] = {ids[int(i)] for i in picks}
                else:                      # 自分の位置以上の添字は 1 つ後ろへずらす
                    self.follows[aid] = {ids[int(i) + 1] if int(i) >= pos else ids[int(i)]
                                         for i in picks}
            self.contacts[aid] = set()
        self.rebuild_follower_index()      # 逆索引を張り直す(乱数ゼロ・値は走査と同一)

    # ---- 被フォロー数の逆索引(純粋な高速化。数える対象・値は旧全走査と完全同一)----
    def rebuild_follower_index(self) -> None:
        """`follows` から逆索引を作り直す(O(辺数)・決定論・乱数ゼロ)。"""
        rev: dict[int, int] = {}
        for targets in self.follows.values():
            for t in targets:
                rev[t] = rev.get(t, 0) + 1
        self._followers = rev

    def _rev(self) -> dict[int, int]:
        """逆索引(古い checkpoint 由来で欠けていれば張り直す)。"""
        rev = getattr(self, "_followers", None)
        if rev is None:
            self.rebuild_follower_index()
            rev = self._followers
        return rev

    def follow(self, follower: int, author: int) -> None:
        """follower が author をフォローする(逆索引も同時更新)。冪等。

        `follows[follower]` が未整備なら空集合から作る(`ensure` を通っていない
        テスト・外部コードの退路)。"""
        f, a = int(follower), int(author)
        bucket = self.follows.setdefault(f, set())
        if a in bucket:
            return
        bucket.add(a)
        rev = self._rev()
        rev[a] = rev.get(a, 0) + 1

    def unfollow(self, follower: int, author: int) -> None:
        """follower が author のフォローを外す(逆索引も同時更新)。冪等。

        ★`follows` を書き換える **6 本目の登録経路**(第117 SNC v2)。用途は
          `net.contact_formation` の `follows_max` 押し出しだけで、シムの中で
          「飽きてフォローを外す」という挙動は**まだ実装していない**(設計書 §2:
          アンフォローは本選後。OASIS も減衰なしで 10 日では影響小)。
        ★索引の減算は `set_follows` と同じ規則(0 になったらキーごと落とす)。
          これで `follower_count` は `_follower_count_scan` と常に一致し続ける。
        """
        f, a = int(follower), int(author)
        bucket = self.follows.get(f)
        if not bucket or a not in bucket:
            return
        bucket.discard(a)
        rev = self._rev()
        n = rev.get(a, 0) - 1
        if n > 0:
            rev[a] = n
        else:
            rev.pop(a, None)

    def drop_contact(self, a: int, b: int) -> None:
        """contacts の相互リンクを外す(第117 SNC v2 の `contacts_max` 押し出し用)。冪等。

        ★`follows` には**触らない**: フォローの上限は別建て(`follows_max`)で管理する
          という設計なので、ここで連鎖して外すと 2 つの上限が絡んで挙動が読めなくなる。
        ★contacts は逆索引を持たない(`follower_count` は follows 側の話)ので、
          ここで壊れる索引は無い。
        """
        x, y = int(a), int(b)
        for lo, hi in ((x, y), (y, x)):
            bucket = self.contacts.get(lo)
            if bucket is not None:
                bucket.discard(hi)

    def set_follows(self, aid: int, targets) -> None:
        """`follows[aid]` を丸ごと差し替える(逆索引も同時更新)。"""
        aid = int(aid)
        rev = self._rev()
        for t in self.follows.get(aid, ()):        # 旧辺を引く
            n = rev.get(t, 0) - 1
            if n > 0:
                rev[t] = n
            else:
                rev.pop(t, None)
        new = {int(t) for t in targets}
        self.follows[aid] = new
        for t in new:
            rev[t] = rev.get(t, 0) + 1

    def ensure(self, aid: int, rng, k: int = 6, candidates: list | None = None) -> None:
        """まだ SNS を持たない個体に初期フォローと空の contacts を据える(冪等・レーン乙 A1)。

        ★なぜ要るか: ``init_follows`` は**起動時 1 回**しか呼ばれないので、プール回転で
          day1 以降に入場する個体は ``follows`` にも ``contacts`` にも載らない。すると
          ``add_contact`` の「双方が contacts に居るときだけ」という条件(下)が永久に
          満たされず、**対面で会話しても DM の相手として登録されない**(= 途中入場者は
          原理的に DM を受け取れない)。タイムラインもフォロー 0 のまま固定される。
        ★分布は ``init_follows`` と同じ「候補から重複なし k 人」。乱数は個体キーの
          **新 named stream** から引く(``RngHub.stream`` は呼ぶたびに新しい Generator を
          派生するので、既存の draw 列は 1 粒も動かない)。
        ★冪等: 既に居る個体には一切触れない(day0 の ``init_follows`` の結果を壊さない)。
        """
        aid = int(aid)
        if aid in self.follows:
            return
        others = candidates or []
        got: set = set()
        if others and int(k) > 0:
            # ★``init_follows`` は ``rng.choice(..., replace=False)`` を使うが、あれは母集団
            #   全体の置換を作るので 25 万人 × 2 万入場では現実的でない。ここは
            #   「整数を引いて重複を捨てる」= 同じ「重複なし k 人の一様抽出」を O(k) で行う
            #   (専用 stream なので他の draw 列には一切影響しない)。
            for _ in range(int(k) * 4):
                if len(got) >= min(int(k), len(others)):
                    break
                cand = int(others[int(rng.integers(len(others)))])
                if cand != aid:
                    got.add(cand)
        self.follows[aid] = got
        self.contacts[aid] = set()
        # ★既読 watermark を「入場時点」に据える(レーンP A5)。据えないと read_marks[aid] は
        #   引くたびに 0 = **街に来る前の全投稿**が未読扱いになり、初回閲覧の
        #   `timeline_for` / `ranked_*` が posts 全履歴を舐める(25 万体・30 日ランでは
        #   途中入場 20.9 万人 × 数十万投稿の走査)。意味の面でも「入場前のタイムラインを
        #   遡って読む」のは現実の挙動ではない(スマホを開いた時点の新着から読む)。
        #   値は `timeline_for` が既読を進めるときと**同じ式**(offset + 件数 = 次に付く post id)。
        #   ★冪等: 既に read_marks を持つ個体(= 再入場・checkpoint 復元)には触れない。
        if aid not in self.read_marks:
            self.read_marks[aid] = self._post_offset + len(self.posts)
        rev = self._rev()                         # 逆索引へ新規辺を足す(O(k))
        for t in got:
            rev[t] = rev.get(t, 0) + 1

    def add_contact(self, a: int, b: int, auto_follow: bool = True) -> None:
        """a と b を知り合いにする(contacts は双方向)。

        `auto_follow=True`(既定 = 現行挙動でバイト不変): さらに a → b の片方向フォローを
        張る(「知り合いは自動フォロー」という初期からの意味づけ)。
        `auto_follow=False`(第117 SNC v2 の C1/C3 が使う): **contacts だけ**を張り、
        `follows` にも `_followers` 逆索引にも 1 バイトも触らない。
        ★理由: 設計 docs/plans/sns-contact-redesign.md §2 は「フォローは**宣言した側だけ**が
          片方向・相互化は自動にしない(全網の相互フォロー率 22.1% を事後検算アンカーにする)」
          と定めている。自動フォローが混ざると『知り合いになった=片方は必ずフォロー』に
          なってしまい、そのアンカーが構造的に測れなくなる。
        """
        if a in self.contacts and b in self.contacts:
            self.contacts[a].add(b)
            self.contacts[b].add(a)
            if not auto_follow:                   # SNC v2: contacts だけ(follows は不触)
                return
            bucket = self.follows[a]              # 知り合いは自動フォロー
            if b not in bucket:
                bucket.add(b)
                rev = self._rev()
                rev[b] = rev.get(b, 0) + 1

    def follower_count(self, author: int) -> int:
        """author をフォローしている人数(= フォロワー数)。Wave G6 のインフルエンサー判定用。

        follows[x] は「x がフォローしている相手」なので逆引きで数える(決定論・乱数なし)。
        ★逆索引 `_followers` を引くだけの **O(1)**(以前は follows 全走査 = O(N) で、
        `hierarchy` の日境界が全員ぶん呼ぶと O(N²) = 25 万人で 1 日 36 分だった)。
        値は `_follower_count_scan`(旧実装)と常に一致しなければならない。"""
        return self._rev().get(int(author), 0)

    def _follower_count_scan(self, author: int) -> int:
        """旧実装(follows 全走査)。**同値性テスト専用の参照実装**で、本番経路は呼ばない。"""
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
        # post は id 一意なので dict の線形 in 判定(O(新着×フォロー)の辞書比較)を
        # id 集合の O(1) 判定に置換(選ばれる集合・順序は完全同一)。
        followed = [p for p in fresh if p["author"] in follows
                    or p["author"] == MEDIA_ID]
        followed_ids = {p["id"] for p in followed}
        rest = [p for p in fresh if p["id"] not in followed_ids]
        prio = self.priority.get(aid)
        if not prio:
            return (followed + rest)[-self.feed_size:]   # 旧ロジック完全一致
        # 優先著者の新着を必ず含める(多すぎれば直近 feed_size 件)。
        prio_posts = [p for p in fresh if p["author"] in prio][-self.feed_size:]
        prio_ids = {p["id"] for p in prio_posts}
        remaining = self.feed_size - len(prio_posts)
        conv = [p for p in (followed + rest) if p["id"] not in prio_ids]
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
