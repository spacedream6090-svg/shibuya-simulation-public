"""SNS・知り合い形成 v2(SNC。**既定 OFF = 現行挙動を 1 バイトも変えない**)。

正典
----
- docs/plans/sns-contact-redesign.md(設計全体・文献根拠・パラメータ・事後検算アンカー)

何が壊れていたか
----------------
現行は `engine/scheduler.py` の発話ハンドラで **「発話が聞こえた全員」** と
`net.add_contact` していた(= 知り合い + 自動フォロー)。半径内に居合わせただけの
傍聴者まで無条件に結ぶので、10k・2 日で follows / contacts が実質**完全グラフ**
(19,738 × 約 19,400 辺)になり RSS 100GiB。25 万では本番が OOM で死ぬ。

「対面で会話した相手」という**意図**は正しく、実装が「聞こえた」を「会話した」と
取り違えていた。Milgram の古典(会話の相手は平均 1.5 人 / 顔見知りは 4 人)に照らすと、
1 発話あたり十数人と知り合うのは桁が 2 つ違う。

本 module が置くもの(4 チャネル + 2 上限)
-------------------------------------------
=========  ============================================================
C1 / F2    **対話成立ペアのみ**。返答者の reply 呼の構造化出力へ二値 2 欄
           (`relate` / `follow`)を足し、LLM の判断で結ぶ。追加 LLM 呼**ゼロ**
           (OASIS = 100 万体 Twitter シムの実運用方式。1 活性化 = 1 呼の
           構造化出力に follow 等の行動選択を同居させる)。
           判定対象は**対話相手に固定**(候補選択なし)なので、「LLM に誰と繋がるか
           選ばせると名前トークンバイアスで網が歪む」という既知の failure mode を
           構造的に回避する。
C3         顔なじみ昇格: 同一相手と `encounter_promote_k` 回目の speak-hear 遭遇で
           挨拶級の contacts(Moreland & Beach: 5 回で有意 / Bornstein: 10-20 回で飽和)。
F3         タイムライン発見: いいねした投稿の著者を確率 `follow_like_p` でフォロー
           (TRF 実測の較正目標 = 新規フォロー 0.03-0.3 件/日/人)。
上限       contacts 200(Dunbar 帯・アクティブ関係 150-250)/ follows 500(安全弁)。
=========  ============================================================

★**O(N²) 再発防止の要**は `encounter_track_max`(既定 100)である。C3 は「相手ごとの
  遭遇回数」を数えるので、素直に書くと **1 個体が全人口ぶんの dict を持つ** = 旧バグと
  同じ O(N²) がメモリ側に再発する。ここは **1 個体あたり `encounter_track_max` 件の
  LRU dict** に固定し、溢れたら最も長く触れていない相手から捨てる(= 忘れる)。
  上限を超えないことと LRU の順序は tests/test_contact_formation.py が機械固定する。
  合計メモリは **O(N × encounter_track_max)** で、N に対して線形。

★`contacts` / `follows` は **Internet 側**(個体側ではない)なのでプール回転の搬送対象外。
  `_encounter_counts` だけが個体側で、これは **transient**(回転で消えてよい = 再入場者は
  顔なじみの数え直しから始まる)。搬送欄を 1 つも増やさない。

R1 ドクトリン
-------------
- 既定 `net.contact_formation.enabled: false` では `enabled()` が False を返し、
  全 seam が即 return = 現行経路(聞こえた全員と add_contact)がそのまま走る。
  L1 バイト一致・プロンプト不変・**乱数を 1 本も引かない**・新 kind 0 件。
- ON でも **generate() の呼び出しサイトは 1 つも増えない**(追加 LLM 呼ゼロ)。
  reply の**出力**が 2 欄ぶん(+20-40 tok)伸びるだけ。
- 乱数は F3 の専用 named stream `"snc_follow"` だけ。`RngHub.stream` は呼ぶたびに
  新しい Generator を派生するので、既存の draw 列は 1 粒も動かない。

正直な限界
----------
1. **知り合い(C1/C3)は follows を 1 本も張らない**。`Internet.add_contact` は既定で
   「知り合いは自動フォロー」だが、SNC 経路は `auto_follow=False` で呼ぶ。理由は
   設計書 §2 F2 の「フォローは**宣言した側だけ**が片方向・相互化は自動にしない」で、
   自動フォローが混ざると事後検算アンカー②(相互フォロー率 ≈ 22.1%)が構造的に
   測れなくなるため。相互フォローは**両者がそれぞれの返事で follow=true を宣言した
   ときだけ**成立する。★SNC OFF の既定経路は従来どおり自動フォローする(バイト不変)。
2. `follows` / `contacts` は **set** なので挿入順が無い = 「最古を押し出し」は原理的に
   復元できない。上限超過の押し出しは決定論の代理規則で行う(下の `_evict_*` の
   docstring に規則と理由を書いた)。上限はどちらも安全弁であって関係の質を測る
   機構ではない(名目 200 / 500 に届く個体は較正が正しければほぼ出ない)。
3. アンフォロー(自然消滅)は本選後(OASIS も減衰なし・10 日では影響小)。
   ここで足す `Internet.unfollow` は**上限の押し出し専用**である。
"""
from __future__ import annotations

from ..observer.schema import Event

SCHEMA = 1

#: L1 の新 kind(★`observer/schema.py` の register_event_kind と
#: `observer/causality.py` の CAUSE_OF_KIND の **2 箇所**に登録済み。片方だけだと
#: 本選 conf の causality ON で `logger.log` が KeyError で即死する)。
KIND_ACQUAINT = "acquaint"
KIND_FOLLOW = "sns_follow"

#: 形成チャネルの札(payload["via"])。設計書 §2 の表と 1 対 1。
VIA_REPLY = "reply"            # C1 / F2 = 返答者の LLM 宣言
VIA_ENCOUNTER = "encounter"    # C3 = 同一相手との k 回目の遭遇
VIA_TIMELINE = "timeline"      # F3 = いいねからの発見

DEFAULTS: dict = {
    "enabled": False,
    "encounter_promote_k": 5,
    "encounter_track_max": 100,
    "contacts_max": 200,
    "follows_max": 500,
    "follow_like_p": 0.02,
}


# --------------------------------------------------------------------------- #
# conf 正準化(既定 OFF)
# --------------------------------------------------------------------------- #
def _get(raw, key, default):
    """OmegaConf / dict どちらでも引ける安全な取り出し(None は既定へ倒す)。"""
    try:
        value = raw.get(key, default)
    except (AttributeError, TypeError):
        return default
    return default if value is None else value


def build_cfg(raw) -> dict:
    """conf の `net.contact_formation` ブロックを型強制つきで正準化する(既定 OFF)。

    数値は下限つきで丸める(0 や負値を渡されて「上限 0 = 全部押し出す」のような
    破壊的な解釈にならないようにする)。`follow_like_p` は [0,1] へクランプ。
    """
    raw = raw if raw is not None else {}
    return {
        "enabled": bool(_get(raw, "enabled", DEFAULTS["enabled"])),
        "encounter_promote_k":
            max(1, int(_get(raw, "encounter_promote_k",
                            DEFAULTS["encounter_promote_k"]))),
        "encounter_track_max":
            max(1, int(_get(raw, "encounter_track_max",
                            DEFAULTS["encounter_track_max"]))),
        "contacts_max":
            max(1, int(_get(raw, "contacts_max", DEFAULTS["contacts_max"]))),
        "follows_max":
            max(1, int(_get(raw, "follows_max", DEFAULTS["follows_max"]))),
        "follow_like_p":
            min(1.0, max(0.0, float(_get(raw, "follow_like_p",
                                         DEFAULTS["follow_like_p"])))),
    }


def _sub(cfg, *path):
    """OmegaConf / dict どちらでも辿れる安全な取り出し(欠けていれば None)。"""
    node = cfg
    for key in path:
        if node is None:
            return None
        try:
            node = node.get(key, None)
        except (AttributeError, TypeError):
            return None
    return node


def cfg_of(sim) -> dict:
    """正準化済み設定を返す。`Simulation` が `netcfg` へ載せた 1 個を使い回す。

    `netcfg` を持たないスタブ sim(テスト・外部コード)では conf から組み直す。
    """
    ncfg = getattr(sim, "netcfg", None)
    if isinstance(ncfg, dict):
        got = ncfg.get("contact_formation")
        if isinstance(got, dict):
            return got
    return build_cfg(_sub(getattr(sim, "cfg", None), "net", "contact_formation"))


def enabled(sim) -> bool:
    """SNC v2 が有効か。既定 OFF。"""
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# プロンプト(reply 用途にだけ足す判定の説明 2 行)
#
# ★no-fingerprint: 機構語(発火・閾値・驚き・モデル・シミュレーション・グラフ・上限)も
#   実験条件語も因子名も 1 文字も書かない。書くのは「返事の JSON に足してよい 2 欄」だけ。
# ★**本文が先・判定が後**(設計書 §1: 構造化出力への二値追加は劣化に強いが、順序で
#   さらに堅くする)。文面もその順で説明する。
# --------------------------------------------------------------------------- #
JUDGMENT_LINES: tuple[str, ...] = (
    "返事の JSON には、本文(text)のあとに relate と follow を真偽値で足してよい。",
    "relate は「この相手と連絡先を交換して知り合いになるか」、"
    "follow は「この相手がSNSに書くことをこれから読みたいか」。"
    "どちらも任意で、そう思わなければ書かなくてよい。",
)


def prompt_section(sim, trigger: str) -> str | None:
    """reply のプロンプトへ足す判定の説明。**OFF / reply 以外は None**(1 行も足さない)。

    ★`prompts.p1` の ON/OFF に依らず同じ 2 行が入る: P1 は reflect/plan/recall の
      ヘッダだけを差し替える機能で、**deliberate(発話・返答)のヘッダは 1 バイトも
      変えない**(prompt_p1.header の分岐)。したがって注入点は 1 つ
      (`deliberate.build_prompt` の reply 枝)で P1 の両経路を覆う。
    """
    if trigger != "reply" or not enabled(sim):
        return None
    return "\n".join(JUDGMENT_LINES)


# --------------------------------------------------------------------------- #
# 上限の押し出し(★`_followers` 逆索引を必ず経由する)
#
# 第116 の不変条件: `follows` を書き換える経路は
#   init_follows / ensure / add_contact / follow / set_follows / **unfollow**
# の 6 つだけで、そのすべてが逆索引 `_followers` を同時に更新する。
# ここは `unfollow`(新設・5 経路規約へ登録済み)だけを使い、`follows[x].discard(y)` の
# ような直接操作は 1 度もしない。
# --------------------------------------------------------------------------- #
def _closeness_key(sim, aid: int, other: int):
    """押し出しの弱さキー(小さいほど先に落ちる)。**決定論・乱数ゼロ**。

    `relations`(Wave G2 の関係台帳)を持っていれば closeness を第一キーにする
    (設計書「closeness 最弱を押し出し」)。台帳が無い / 相手の行が無い場合は
    `set` に挿入順が無いので FIFO は復元できない。代理として
    「最終接触 step の古い順 → id 昇順」を使う(= 最も長く音沙汰の無い相手)。
    値が全部欠測なら id 昇順の決定論に落ちる。
    """
    agent = None
    by_id = getattr(sim, "agent_by_id", None)
    if by_id is not None:
        agent = by_id.get(int(aid))
    rel = None
    if agent is not None:
        mem = getattr(agent, "mem", None)
        rels = getattr(mem, "relations", None)
        if isinstance(rels, dict):
            rel = rels.get(int(other))
    if not isinstance(rel, dict):
        return (0.0, -1, int(other))
    return (float(rel.get("closeness", 0.0)),
            int(rel.get("last_step", -1)), int(other))


def _evict_contacts(sim, aid: int, cfg: dict, keep: int) -> None:
    """`contacts[aid]` を上限まで削る(closeness 最弱から。`keep` は落とさない)。

    contacts は**双方向**なので相手側の集合からも同時に外す(片側だけ残ると
    「自分は DM できないが相手からは来る」という非対称が生まれる)。
    follows には触らない(別の上限で管理する = 上限どうしを絡ませない)。
    """
    net = sim.net
    bucket = net.contacts.get(int(aid))
    if bucket is None:
        return
    cap = int(cfg["contacts_max"])
    while len(bucket) > cap:
        victims = [t for t in bucket if t != int(keep)]
        if not victims:
            break
        victim = min(victims, key=lambda t: _closeness_key(sim, aid, t))
        net.drop_contact(int(aid), int(victim))


def _evict_follows(sim, aid: int, cfg: dict, keep: int) -> None:
    """`follows[aid]` を上限まで削る(`keep` は落とさない)。**必ず unfollow 経由**。

    ★正直な限界: `follows` は set なので「最古」は原理的に復元できない。決定論の
      代理規則として **contacts に居ない相手を先に**落とし、その中で id 昇順とする
      (= 知り合いの購読より、顔も知らない相手の購読を先に手放す)。名目 500 の
      安全弁なので、較正が正しければここに到達する個体はほぼ出ない。
    """
    net = sim.net
    bucket = net.follows.get(int(aid))
    if bucket is None:
        return
    cap = int(cfg["follows_max"])
    contacts = net.contacts.get(int(aid), ())
    while len(bucket) > cap:
        victims = [t for t in bucket if t != int(keep)]
        if not victims:
            break
        victim = min(victims, key=lambda t: (1 if t in contacts else 0, int(t)))
        net.unfollow(int(aid), int(victim))


# --------------------------------------------------------------------------- #
# 形成の 2 つの原子操作(L1 記録 + 上限の押し出しまで込み)
# --------------------------------------------------------------------------- #
def _log(sim, actor, kind: str, step: int, sim_min: int, payload: dict) -> None:
    logger = getattr(sim, "logger", None)
    if logger is None:                             # スタブ sim(純関数テスト)への退路
        return
    logger.log(Event(step=step, sim_min=sim_min, agent_id=int(actor.id),
                     kind=kind, x=getattr(actor, "x", 0.0),
                     y=getattr(actor, "y", 0.0), payload=payload))


def acquaint(sim, actor, other_id: int, via: str, step: int, sim_min: int) -> bool:
    """actor と other を知り合いにする(双方向)。成立したら L1 へ 1 件残し True。

    既に知り合いなら**何もしない**(冪等・イベントも出さない = 同じ縁を毎日数えない)。
    形成は `Internet.add_contact` に委ねるが、**`auto_follow=False`**(= contacts だけ・
    follows には 1 バイトも触らない)で呼ぶ。

    ★なぜ自動フォローを切るか(設計書 §2 F2): フォローは「**宣言した側だけ**が片方向」で
      あり、相互化は自動にしない。全網の相互フォロー率 **22.1%**(Kwak)を事後検算の
      アンカーにするので、「知り合いになった = 片方は必ずフォロー」が混ざるとその量が
      構造的に測れなくなる。したがって:
        C1(relate) → contacts 双方向のみ
        C3(遭遇昇格) → contacts 双方向のみ
        F2(follow) → 返答者 → 話者の片方向フォローだけ(下の `follow()`)
      相互フォローは **両者がそれぞれの返事で follow=true を宣言したときだけ**成立する。
    ★follows を 1 本も足さないので `follows_max` の押し出しもここでは要らない。
    """
    net = sim.net
    aid, oid = int(actor.id), int(other_id)
    if aid == oid:
        return False
    if oid in net.contacts.get(aid, ()):           # 既知 = 冪等
        return False
    if aid not in net.contacts or oid not in net.contacts:
        return False                               # 片方が SNS 未配線(add_contact と同条件)
    net.add_contact(aid, oid, auto_follow=False)
    cfg = cfg_of(sim)
    _evict_contacts(sim, aid, cfg, keep=oid)
    _evict_contacts(sim, oid, cfg, keep=aid)
    _log(sim, actor, KIND_ACQUAINT, step, sim_min, {"other": oid, "via": via})
    return True


def follow(sim, actor, author_id: int, via: str, step: int, sim_min: int) -> bool:
    """actor が author を**片方向**フォローする。成立したら L1 へ 1 件残し True。

    既にフォロー済みなら何もしない(冪等)。相互化は自動でしない(設計書 §2 F2:
    全網の相互フォロー率 22.1% を事後検算アンカーにするため、こちらから相互化しない)。
    """
    net = sim.net
    aid, author = int(actor.id), int(author_id)
    if aid == author or author < 0:                # 自分・メディア(-1)は対象外
        return False
    if author in net.follows.get(aid, ()):         # 既知 = 冪等
        return False
    net.follow(aid, author)
    _evict_follows(sim, aid, cfg_of(sim), keep=author)
    _log(sim, actor, KIND_FOLLOW, step, sim_min, {"author": author, "via": via})
    return True


# --------------------------------------------------------------------------- #
# C3: 顔なじみ昇格(★O(N²) 再発防止の要 = 1 個体あたり LRU 有界)
# --------------------------------------------------------------------------- #
def note_encounter(sim, hearer, speaker, step: int, sim_min: int) -> None:
    """speak-hear の遭遇 1 件を数え、k 回目で挨拶級の contacts へ昇格する。

    ★数えるのは**聞き手側の台帳**(聞き手 → 話者)。話者は 1 発話で十数人と遭遇するが、
      聞き手から見れば「その話者と 1 回すれ違った」= 対称に 1 件ずつ増える。

    ★有界性(この関数が新しい O(N²) にならない根拠):
      ① 既に知り合いの相手は**数えない**(即 return)= 昇格済みの縁は台帳を占有しない。
      ② `_encounter_counts` は 1 個体あたり最大 `encounter_track_max` 件の LRU dict。
         触れた相手を末尾へ移し、溢れたら先頭(= 最も長く触れていない相手)を捨てる。
         Python の dict は挿入順を保つので `pop` → 再代入で O(1) の LRU になる。
      ③ よって全体メモリは O(N × encounter_track_max) = N に対して**線形**。
      ④ `_encounter_counts` は個体側 transient(プール回転で消えてよい)。搬送欄なし。
    """
    cfg = cfg_of(sim)
    hid, sid = int(hearer.id), int(speaker.id)
    if hid == sid:
        return
    if sid in sim.net.contacts.get(hid, ()):       # ① 既に知り合い = 数える意味がない
        return
    counts = getattr(hearer, "_encounter_counts", None)
    if counts is None:
        counts = {}
        hearer._encounter_counts = counts
    n = counts.pop(sid, 0) + 1                     # ② pop → 再代入 = 末尾へ(LRU touch)
    counts[sid] = n
    track_max = int(cfg["encounter_track_max"])
    while len(counts) > track_max:
        counts.pop(next(iter(counts)))             # 先頭 = 最も長く触れていない相手
    if n >= int(cfg["encounter_promote_k"]):
        # 昇格**できたときだけ**台帳から降ろす(①と同じ理由 = 既知の相手で枠を食わない)。
        # 相手が SNS 未配線などで結べなかった場合は数え続け、次の遭遇で再試行する
        # (捨ててしまうと「あと k 回会わないと再挑戦できない」という不自然が残る)。
        if acquaint(sim, hearer, sid, VIA_ENCOUNTER, step, sim_min):
            counts.pop(sid, None)


# --------------------------------------------------------------------------- #
# C1 + F2: 返答の LLM 宣言(追加 LLM 呼ゼロ)
# --------------------------------------------------------------------------- #
def on_reply(sim, replier, speaker_id: int, action, step: int, sim_min: int) -> None:
    """返答 1 本の構造化出力から `relate` / `follow` を読み、関係を結ぶ。

    **欠落 = false**(安全側)。mock バックエンドは 2 欄を出さないので mock ランでは
    C1/F2 は 1 件も成立しない = 「LLM が言っていないのに縁ができる」ことは起きない。

    - relate=true → `net.add_contact(話者, 返答者)`(双方向・現行 API)
    - follow=true → 返答者 →**片方向**→ 話者(`net.follow`)

    ★返答イベントの因果の中(= 同じ step / sim_min・同じ個体の返答の直後)で処理する。
    """
    if not enabled(sim) or not isinstance(action, dict):
        return
    if speaker_id is None or int(speaker_id) < 0:
        return
    speaker = None
    by_id = getattr(sim, "agent_by_id", None)
    if by_id is not None:
        speaker = by_id.get(int(speaker_id))
    if speaker is None:                            # 名簿から引けない = 何もしない
        return
    if action.get("relate") is True:
        # 現行 API の引数順に合わせて (話者, 返答者)。contacts は双方向なので
        # どちらを actor にしても縁は同じだが、L1 の agent_id は「話者」になる。
        acquaint(sim, speaker, int(replier.id), VIA_REPLY, step, sim_min)
    if action.get("follow") is True:
        follow(sim, replier, int(speaker_id), VIA_REPLY, step, sim_min)


# --------------------------------------------------------------------------- #
# F3: タイムライン発見(いいね → 確率フォロー)
# --------------------------------------------------------------------------- #
def on_like(sim, agent, post: dict, step: int, sim_min: int) -> None:
    """いいねした投稿の著者を確率 `follow_like_p` でフォローする。

    ★乱数は用途別の**新 named stream** `"snc_follow"` を (agent, step, post) で引く。
      `RngHub.stream` は呼ぶたびに新しい Generator を派生するので、既存の draw 列
      (phone / sns_react …)は 1 粒も動かない。post id をキーに含めるのは、1 回の閲覧で
      複数の投稿へいいねしたときに**同じ乱数を使い回さない**ため(stream は毎回
      同じ種から作り直されるので、キーが同じなら 1 本目の draw も同じになる)。
    ★**OFF では 1 粒も引かない**(stream を作りもしない)= バイト一致。
    """
    if not enabled(sim):
        return
    author = int(post.get("author", -1))
    if author < 0 or author == int(agent.id):      # メディア(-1)・自分の投稿は対象外
        return
    if author in sim.net.follows.get(int(agent.id), ()):
        return                                     # 既にフォロー済み = 乱数も引かない
    cfg = cfg_of(sim)
    p = float(cfg["follow_like_p"])
    if p <= 0.0:
        return
    rng = sim.hub.stream("snc_follow", int(agent.id), int(step),
                         int(post.get("id", 0)))
    if rng.random() < p:
        follow(sim, agent, author, VIA_TIMELINE, step, sim_min)
