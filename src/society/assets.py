"""所有権レイヤー O1(登記簿)+ O3(相続)= 「所有されるもの」を世界の第一級オブジェクトに
する層(``world.assets``・**既定 OFF**)。

正典
----
- ``docs/plans/ownership-layer-plan.md`` **§1**(AssetLedger = 権利行 + 双方向索引 /
  資産保存則 = 貨幣保存則 IF-E の完全な双対 / 階層 LoD)・**§2 の O1 と O3 行**・
  **§5 ユーザー決定 3 件**(①住戸の初期所有者に**域内の不動産会社も加える** /
  ②相続 = 承認・O1 と同時実装 / ③本線前に実装)
- ``docs/research/ownership-asset-models.md`` §4-1(4 方式は排他ではなくレイヤが違う)・
  §4-2(行を「所有」ではなく**権利**にする = LADM の RRR / bundle of rights)・
  §4-3(移転の統一語彙と資産保存則)・§5-2(**推奨 = 「タグ付きレコードを登記簿に置く」**)

何を解く問題か
--------------
本リポジトリの「所有」は、**金だけが完全**で、物と権利が空白である(研究文書 §2 の棚卸し):

  - 建物は地物で、**土地・建物の所有者が世界に一人も居ない**。家賃・敷金の受け手は
    「不在家主」として RoW へ落ちる(``scheduler`` の ``rent_landlord`` /
    ``deposit_landlord`` チャネル)。
  - 「空き住戸」は登記簿が無いので「**他エージェントの home でない**」という否定条件でしか
    書けない(``freedom_p2.pick_home`` の全 agent 走査 = 登記簿不在の実費)。
  - ``has_car`` は bool タグで、個体性も移転も無い(リポ最古の所有表現)。
  - **相続が存在しない**: ``health._die`` は ``dead=True`` にするだけで、死者の ``money`` は
    エージェントオブジェクトに凍結されたまま = Σ 不変量の上では「死者の財布」という
    暗黙部門が生まれている(会計上の漏れではないが、**観測上の盲点**)。

本 module は、``lost_property`` が**事件のために**作った資産レコード(owner タグ付き item を
中央 dict に置く形)を、**平時の所有一般へ昇格**させる。ゼロから作る話ではない。

方式(ユーザーの「物にタグ」案への回答 = ハイブリッド)
------------------------------------------------------
ユーザー案の核心「所有されているもの自体に id と属性を持たせて観察・管理する」は正しい。
ただし**タグをどこに置くか**で長期費用が分かれる(研究文書 §5-2)。本 module は

    **タグ付きレコードを登記簿に置く**(Torrens の「台帳が真実」+ 履歴は L1)

を採る。すなわち資産レコードは owner/right のタグを持ち、**置き場と書き手は中央台帳 1 つ**に
限定する(``sim._ledger_state`` / 単一の書き手 :func:`transfer`)。これで

  - 逆引き(この住戸は誰の物か)が O(1)(``by_asset``)・順引き(この人の持ち物)も O(1)(``by_owner``)
  - 資産保存則の Σ 検査が**台帳 1 箇所**で書ける(分散したタグを全走査しない)
  - OFF では**台帳オブジェクトを作らない**だけで完結する(属性を agent に生やすと
    checkpoint に載り、OFF の不侵襲性が崩れる)

の 3 つが同時に立つ。IFC の ``IfcOwnerHistory``(全オブジェクトに所有タグ)が必須 → 任意 →
廃止提案と辿った実運用の轍(研究文書 §3 #12)を踏まない形でもある。

行 = 権利(RRR)
----------------
行は「所有」ではなく**権利** ``(asset_id, party_id, right_kind, since)`` にする(LADM の
Party-RRR-BAUnit / Honoré の権利の束)。「賃貸中」= 同じ資産に ``own``(家主)と ``lease``(借主)が
並ぶ状態、``venture`` の ``permitted`` bool は ``permit`` 行の有無、遺失物の ``hold`` は
「``own`` は元 owner のまま ``custody`` が無い」状態 —— すべて同じスキーマの特殊例になる。

★ただし **O1 の既定 ON の最小核は ``own`` 1 種だけ**。``right_kinds`` は conf に列挙を宣言して
段階拡張する(IF-B の 3 水準 conf 化と同じ流儀)。lease/permit/lien/custody の行化は O4。

party(主体)は agent / org / RoW(域外の家主・本社)の 3 型で、IF-E2 の部門語彙をそのまま使う。
**個人所有と組織所有は party 型の違いだけ**で、機構は 1 つである(ユーザー要件 R-c)。

資産保存則(貨幣保存則 IF-E の完全な双対)
-------------------------------------------
「台帳の行は **RoW(製造・輸入)からのみ生まれ、K5(廃棄・滅失)でのみ消え、それ以外の全変化は
party の付け替えである**」(SNA 2008 の資産変動三分法 = 取引 / 境界フロー / K.5)。

    Σ(所有者別保有数) + K5 累積 − RoW 生成累積 = 初期ストック   (カテゴリ別)

★O1 では**製造も廃棄も起こさない**ので ``born`` / ``k5`` は常に 0 = 生存行数が初期ストックに
  張り付く。テストはその強い形で固定する(将来 O2 の耐久財寿命が入ると両項が動き出す)。
移転語彙は 9 種に正規化し(:data:`TRANSFER_KINDS`)、**表に無い種は捨てずに ``unclassified`` へ
数える**(IF-E の未分類監視装置の双対 = 新しい移転経路が入った瞬間に鳴る)。

規模 = 階層 LoD
---------------
25 万 agent × 家財数十点 = 数百万レコードは dict では成立しない(研究文書 §6)。答えは階層:

  - **L-full(登記簿の行)** … 不動産(住戸)・車両。地図 v8 の住宅系建物は 5,531 棟・
    延べ階数 11,948 = **1 万行の桁**。車両は ``has_car`` 8% の個体昇格。合計しても既存 org 台帳
    (9,872 社)と同じ桁で、現行の dict 流儀のままで足りる。
  - L-agg(家電・家具 = 世帯集計 + イベント時だけ個体化)… **O2**(本 module では作らない)
  - L-flow(食料・消耗品 = 現行 spend のまま)… SNA も消耗品は資産境界の外

「範囲内のあらゆるもの」は「**あらゆるものが観測に応答できる**」ことで満たす(MATSim の
粒度原則)。将来ほんとうに全個体が要るときは ``engine/soa.py`` の SoA 列へ**格納だけ**
差し替える(意味論は不変)。

自然主義との整合(売買・贈与を**勧めない**)
--------------------------------------------
本 module は**新しい行動候補を 1 つも作らない**。既存の行為(死亡・転居)が台帳に写像される
だけの「分類の追加」(IF-E2 UNCOVERED 方式)から始める。「自分の持ち物」をプロンプトに
見せる段・中古売買を選べるようにする段は独立トグル(affects_k 再評価つき)= O5。

R1 ドクトリン
-------------
- 既定 ``world.assets.enabled: false`` では :func:`arm` / :func:`phase` が即 return し、
  **台帳オブジェクト自体を作らない**(``sim._ledger_state`` が生えない・L1 に 1 件も出ず・
  agent に属性が生えず・プロンプトが 1 バイトも変わらず・乱数 stream を 1 本も引かない)。
  ★とくに**死者の財布は現行どおり凍結されたまま**= 第107 と完全同値。
- **新乱数は新 named stream ``"asset_alloc"`` 1 本だけ**(初期配賦の住戸 1 戸 1 draw)。
  家主 org の選定・相続の分配は **乱数を 1 粒も引かない**(sha256 の純関数ジッタと id 昇順)。
- ``generate()`` の呼び出しサイトを 1 つも作らない(**LLM 呼数増分ゼロ** = k 非依存)。
  第 1 段は**観測層に閉じる**ので、当事者の記憶にも 1 行も入れない(プロンプト完全不変)。
- 毎 step の全資産走査はゼロ。走るのは (a) 起動時 1 回の初期配賦 と
  (b) **新しい L1 の watermark 走査**(死亡・転居があった step だけ仕事をする)。

較正(数字の出どころ)
----------------------
- **持ち家住宅率 32%**: 総務省統計局「住宅・土地統計調査」の渋谷区の持ち家住宅率
  (東京都全体 ≈45% / 特別区部 ≈43% に対し、渋谷区は借家が多く 3 割台前半)。
  ★本値は**桁と順序の較正アンカー**であって、調査年による揺れ(±3pt 程度)は
  ``world.assets.owner_occupancy_rate`` で動かせる。「渋谷は借家の街」= 住戸の 2/3 以上が
  賃貸、が本 module の較正の主張である。
- **賃貸住戸の家主の 2 割が法人**: 国土交通省「賃貸住宅管理業務に関する実態調査」系の
  「賃貸住宅所有者の約 8 割が個人」から、法人所有 ≈ 2 割を採る。本シムには「域外の個人家主」を
  表す主体が RoW しか無いので、**法人 = 域内の不動産 org / 個人 = RoW(域外家主)** と写す
  (``org_landlord_share`` で調整可)。これがユーザー決定 §5-1「域内の不動産会社も所有者に
  追加する」の実装形である。
- **域内不動産 org の特定**: 組織台帳(``data/organizations_shibuya_census.json``・9,872 社)の
  ``industry_key == "RE"``(不動産業・物品賃貸業 = 996 社)のうち、``sector_detail`` が
  **賃貸仲介/管理(256 社)・開発/PM(231 社)** のものを家主の担い手とする(合計 487 社)。
  オフィス仲介(258)と物品賃貸(251)は**住戸の家主ではない**ので外す = 台帳に実在する
  区分だけで決まる、仮定の要らない特定である。台帳に該当が 1 社も無い版(旧 wide 台帳など)
  では ``office`` カテゴリの決定論部分集合へ後退し、``org_source`` にその旨を**正直に残す**。
- **家主 org の中の配り方**は従業者数(``size.employees``)の重み付き(大きい会社ほど多く
  持つ)。均等割りにすると所有ネットワークが退化して集中度の観測(研究文書 §7-2)が死ぬ。

正直な限界(7 件)
------------------
1. **売買代金は動かさない**。持ち家者の転居に伴う ``own`` 移転(:data:`SALE`)は**台帳上の
   分類だけ**で、金は 1 円も動かさない(挙動 1 バイト不変の原則)。「金と物が同時に動く」
   突合(研究文書 §7-5)が成立するのは中古売買を内生化する O5 からである。
2. **家賃・敷金の受け手を付け替えない**。``rent_landlord`` / ``deposit_landlord`` は今も RoW へ
   落ちる。台帳上の家主が域内 org でも、その org の預金は 1 円も増えない(= 金の側は未接続)。
   これも O4(lease 行)+ O5 の仕事で、ここで繋ぐと家賃の会計が二重になる。
3. **初期配賦は step 0 の在場者を基準にする**。プール回転(100 万プール)で後から街へ入る
   個体は「既に誰かの物になっている住戸」に住むことになる。ユーザー決定「所有者はラン中静的」
   と整合する簡約だが、プール ON では「持ち家に住んでいるのに own 行が別人」の個体が出る。
4. **住戸の粒度は (建物, 階)**。実際のマンションは 1 フロアに複数住戸があるが、世界の側が
   ``home_building`` × ``home_floor`` までしか持っていないので、それ以上細かくすると
   台帳だけが実在しない粒度を持つことになる(捏造しない)。
5. **``inheritance`` は ``analyze_accounting`` の ``MONEY_KINDS`` にまだ足していない**ので、
   同スクリプトの ``unclassified_money_kinds`` に「新しい金の経路」として**正直に列挙される**
   (= 監視装置が設計どおり鳴っている状態)。部門行列に載せるには家計 → 家計の世代間移転と
   「相続人不存在 = 国庫」の扱いを決める必要があり、それは会計側の設計判断なので本レーンでは
   踏み込まない(``lost_property`` の ``lost_return`` / ``lost_keep`` / ``lost_expire`` と同じ線引き)。
6. **事業持分(venture / org)は台帳に載せない**。O1 のカテゴリは不動産と車両の 2 つだけで、
   ``venture`` の開閉は写像しない(計画書 §2 の O1 行が不動産 + 車両に限っているため)。
7. **街に居ない世帯員は相続人になれない**。プール回転で退場中の個体は脱水されていて残高を
   書けないので、配偶者が居ても「その日たまたま街に居なかった」なら遺産は国庫へ出る。
   黙って落とすと観測から消えるので、その人数を L1 の ``absent`` と
   ``summary.assets.heirs_absent`` に**必ず載せる**(縦煙の実測: 300 体・在場のみの構成で
   死 3 件のうち 2 件が absent=1 で国庫行き)。受け皿を広げるには pool の再入場と
   相続を結ぶ必要があり、それは pool レーンの設計判断なのでここでは踏み込まない。
   ★レーン甲(2026-08-13): この「居ない」の**判定そのものが壊れていた**。``agent_by_id`` は
   退場者を消さない索引なので ``get() is None`` は決して真にならず、実際には**不在の相続人の
   幽霊オブジェクトへ残高を書いていた**(hydrate で捨てられ、故人の側だけ 0 になる = 現金消失)。
   いまは ``sim.present_agent``(在場述語)で解決するので、上の記述が初めて実装と一致する。
"""
from __future__ import annotations

import hashlib

from . import economy_sfc as sfc_mod
from .observer.schema import Event, register_event_kind

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種(**材料側 registration**: lost_property.py / city_ops.py / traces.py と
# 同じ流儀 = observer/schema.py 本体を触らずに自分の種を自分で登録する)。
# ★payload に自由文は 1 つも入らない(カテゴリ・移転語彙は下の有限表の語・party は
#   世界の識別子・残りは件数と金額)。
# --------------------------------------------------------------------------- #
register_event_kind("inheritance", "★死亡による資産と現金の世帯内移転(O3)。"
                                   "agent_id = 故人"
                                   "{heirs, assets, amount, to}")
register_event_kind("asset_transfer", "登記簿の権利行が別の party へ移った(売買・贈与ほか)。"
                                      "agent_id = 当事者(居なければ -1)"
                                      "{asset, cat, right, from, to, kind}")


# --------------------------------------------------------------------------- #
# 語彙(**ここに無い語は存在しない**)
# --------------------------------------------------------------------------- #
#: 権利種(LADM の RRR)。**O1 が書くのは own 1 種だけ**で、残りは将来枠(O4)の宣言。
OWN = "own"
LEASE = "lease"        # 賃借(O4)
PERMIT = "permit"      # 許認可(O4。venture の permitted bool の一般化)
LIEN = "lien"          # 担保(O4)
CUSTODY = "custody"    # 占有(O4。拾得者・在院の一般化)
RIGHT_KINDS: tuple[str, ...] = (OWN, LEASE, PERMIT, LIEN, CUSTODY)

#: 資産カテゴリ(L-full = 登記簿に全個体で載せる層だけ)。
DWELLING = "dwelling"  # 住戸 =(住宅系建物, 階)
VEHICLE = "vehicle"    # 車両 = has_car の個体昇格
CATEGORIES: tuple[str, ...] = (DWELLING, VEHICLE)

#: party の型(IF-E2 の部門語彙と同じ 3 型)。id は前置詞つきの文字列で名前空間を分ける。
P_AGENT = "a"          # 個人(agent id)
P_ORG = "o"            # 組織(org 台帳の id)
P_ROW = "row"          # rest-of-world(域外の家主・域外資本)
PARTY_KINDS: tuple[str, ...] = ("agent", "org", "row")

#: 移転語彙(研究文書 §4-3 の 9 種に正規化)。**表に無い種は ``unclassified`` へ数える**。
SALE = "sale"                  # 売買(持ち家者の転居 = 家主へ売却)
GIFT = "gift"                  # 贈与(O5)
INHERIT = "inheritance"        # 相続(O3)
ESCHEAT = "escheat"            # 相続人不存在 → 国庫(民法 959 条)= 街の外 = RoW
LEASE_OUT = "lease"            # 貸借(O4)
PLEDGE = "lien"                # 担保(O4)
LOST = "lost"                  # 遺失/拾得(既存 lost_property の一般化・O4)
THEFT = "theft"                # 窃盗(own 不変・custody だけ移る = 法的に正確。O4)
BORN = "born"                  # 製造・輸入(源 = RoW。O2)
SCRAP = "scrap"                # 廃棄・滅失(先 = K5。O2)
TRANSFER_KINDS: tuple[str, ...] = (SALE, GIFT, INHERIT, ESCHEAT, LEASE_OUT, PLEDGE,
                                   LOST, THEFT, BORN, SCRAP)

#: 初期配賦の所有者内訳のラベル(provenance に出す観測量)。
ALLOC_RESIDENT = "resident"    # 居住者自身(持ち家)
ALLOC_ORG = "org"              # 域内の不動産 org(ユーザー決定 §5-1)
ALLOC_ROW = "row"              # 域外の家主(不在家主)


# --------------------------------------------------------------------------- #
# 較正済みの既定値(**すべて module docstring の「較正」節に出どころがある**)
# --------------------------------------------------------------------------- #
#: 渋谷区の持ち家住宅率(住宅・土地統計調査。都全体 ≈45% / 特別区部 ≈43% に対し 3 割台前半)。
OWNER_OCCUPANCY_RATE: float = 0.32

#: 賃貸住戸の家主のうち**法人**の割合 ≈ 2 割(個人 8 割)→ 域内の不動産 org に写す。
ORG_LANDLORD_SHARE: float = 0.20

#: 域内不動産 org の特定鍵(組織台帳の実在フィールド。docstring「較正」節)。
LANDLORD_INDUSTRY_KEY: str = "RE"
LANDLORD_SECTORS: tuple[str, ...] = ("賃貸仲介/管理", "開発/PM")

DEFAULTS: dict = {
    "enabled": False,
    # ---- 権利種の宣言(O1 が書くのは own 1 種。残りは段階拡張の枠)----
    "right_kinds": (OWN,),
    # ---- 初期配賦 ----
    "owner_occupancy_rate": OWNER_OCCUPANCY_RATE,
    "org_landlord_share": ORG_LANDLORD_SHARE,
    "landlord_industry_key": LANDLORD_INDUSTRY_KEY,
    "landlord_sectors": LANDLORD_SECTORS,
    "max_landlord_orgs": 512,     # 家主にする org の上限(決定論部分集合 = id 昇順の先頭)
    "dwellings": True,            # 住戸を台帳に載せるか(カテゴリ別の入切)
    "vehicles": True,             # 車両を台帳に載せるか
    # ---- 安全弁 ----
    "max_assets": 400_000,        # 台帳の行数の上限(超えたら enumerate を打ち切る)
    "max_events_per_step": 32,    # 1 step に出す本 module の L1 件数の上限
}

_FLOAT_KEYS = ("owner_occupancy_rate", "org_landlord_share")
_INT_KEYS = ("max_landlord_orgs", "max_assets", "max_events_per_step")
_BOOL_KEYS = ("dwellings", "vehicles")
_TUPLE_KEYS = ("landlord_sectors",)
_STR_KEYS = ("landlord_industry_key",)


# --------------------------------------------------------------------------- #
# cfg 正準化(lost_property.build_cfg / traces.build_cfg と同型: dict / OmegaConf 両対応)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の ``world.assets`` ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るので型強制する(``traces.build_cfg`` と同じ作法)。
    ``right_kinds`` は**有限表 :data:`RIGHT_KINDS` に無い語を黙って捨て**、``own`` は必ず
    先頭に置く(登記簿の最小核が config で消せてしまうと台帳が空回りするため)。
    比率は [0, 1] にクリップする(退化と符号反転の防止)。
    """
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    for key, val in raw.items():
        if key == "enabled":
            cfg["enabled"] = (val if isinstance(val, bool)
                              else str(val).strip().lower() == "true")
        elif key in _BOOL_KEYS:
            cfg[key] = (val if isinstance(val, bool)
                        else str(val).strip().lower() == "true")
        elif key in _FLOAT_KEYS:
            cfg[key] = min(1.0, max(0.0, float(val)))
        elif key in _INT_KEYS:
            cfg[key] = max(0, int(val))
        elif key in _STR_KEYS:
            cfg[key] = str(val)
        elif key in _TUPLE_KEYS:
            cfg[key] = tuple(str(s) for s in (_to_plain(val) or ()))
        elif key == "right_kinds":
            got = [str(s) for s in (_to_plain(val) or ()) if str(s) in RIGHT_KINDS]
            cfg["right_kinds"] = tuple(
                [OWN] + [k for k in RIGHT_KINDS if k != OWN and k in got])
    return cfg


def cfg_of(sim) -> dict:
    """所有権レイヤー設定(初回のみ ``sim.cfg.world.assets`` から遅延構築してキャッシュ)。

    simulation.py は読み取り専用なので cfg は本 module が遅延構築する
    (``lost_property.cfg_of`` / ``economy_sfc.cfg_of`` と同型)。キャッシュ属性
    ``sim.ledgercfg`` は L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。
    """
    c = getattr(sim, "ledgercfg", None)
    if c is None:
        try:
            c = build_cfg((sim.cfg.get("world", None) or {}).get("assets", None))
        except Exception:                          # noqa: BLE001(旧 config 互換)
            c = build_cfg(None)
        sim.ledgercfg = c
    return c


def enabled(sim) -> bool:
    """登記簿が有効か。**既定 OFF = 本 module は 1 バイトも世界に触らない**。"""
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 決定論ジッタ(**乱数 stream を 1 本も消費しない純関数**)
#
# ``lost_property._jitter`` / ``city_ops._collapse_gate`` と同じハッシュゲート。
# (master_seed, 用途, キー…) の sha256 から [0,1) を作る。同 seed なら resume を跨いでも
# 編成順が変わっても同じ値になる。
# --------------------------------------------------------------------------- #
def _jitter(seed: int, *parts) -> float:
    blob = "/".join([str(int(seed))] + [str(p) for p in parts])
    digest = hashlib.sha256(blob.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") / float(1 << 64)


def _seed_of(sim) -> int:
    try:
        return int(sim.hub.master_seed)
    except Exception:                              # noqa: BLE001(hub 不在の単体テスト)
        return 0


# --------------------------------------------------------------------------- #
# party / asset の識別子(**文字列 1 本で名前空間を分ける**)
# --------------------------------------------------------------------------- #
def agent_party(agent_id) -> str:
    return f"{P_AGENT}:{int(agent_id)}"


def org_party(org_id) -> str:
    return f"{P_ORG}:{org_id}"


def row_party() -> str:
    return P_ROW


def party_kind(party: str) -> str:
    """party id → 型("agent" / "org" / "row")。未知の綴りは "row" に寄せない(捏造しない)。"""
    p = str(party)
    if p == P_ROW:
        return "row"
    if p.startswith(P_AGENT + ":"):
        return "agent"
    if p.startswith(P_ORG + ":"):
        return "org"
    return "unknown"


def party_agent_id(party: str) -> int | None:
    """party が個人ならその agent id(そうでなければ None)。"""
    p = str(party)
    if not p.startswith(P_AGENT + ":"):
        return None
    try:
        return int(p.split(":", 1)[1])
    except ValueError:
        return None


def dwelling_id(building: str, floor: int) -> str:
    return f"dw:{building}:{int(floor)}"


def vehicle_id(agent_id) -> str:
    return f"vh:{int(agent_id)}"


# --------------------------------------------------------------------------- #
# 登記簿(ON 経路でのみ生やす。checkpoint.py が runtime["asset_ledger"] で中央管理する)
#
#   assets[aid]   … 資産レコード {"cat", "building", "floor", "node", "agent"}
#   rows[aid]     … 権利行 {right_kind: {"party", "since"}}(**by_asset 索引そのもの**)
#   by_owner[pid] … {aid: right_kind}(**逆向きの索引**。O(1) で「この人の持ち物」が引ける)
#   stock0[cat]   … 初期ストック(資産保存則の右辺)
#   born/k5[cat]  … 製造・輸入 / 廃棄・滅失の累積(O1 では常に 0)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_ledger_state", None)
    if st is None:
        st = {
            "schema": SCHEMA,
            "init": False,
            "assets": {},
            "rows": {},
            "by_owner": {},
            "stock0": {},
            "born": {},
            "k5": {},
            # ★境界流入(A12)= プール回転で**街に入ってきた既存資産**の累積。
            #   「後日判明した初期条件」として stock0 に足すと保存則の右辺が走行中に動く
            #   (= 検査が「あとから辻褄を合わせる装置」に堕ちる)ので、恒等式の**流入側**に
            #   独立の項として立てる(analyze_accounting の rot_in / rot_out と同型の帳尻項)。
            "rot_in": {},
            "veh_day": 0,             # 車両の日次追随を最後に走らせた day(冪等ガード)
            "transfers": {},
            "unclassified": {},
            "alloc": {},              # 初期配賦の所有者内訳(resident / org / row)
            "landlords": [],          # 家主にした域内 org の id(決定論・観測用)
            "org_source": "none",     # 域内不動産 org の特定方法(正直な宣言)
            "capped": False,          # max_assets で打ち切ったか
            "heired": [],             # 相続済みの故人 id(冪等ガード。sorted list で保存)
            "deaths": 0,
            "heirs_absent": 0,        # 街に居なかった世帯員の累計(手が届かなかった相続人)
            "inherited_assets": 0,
            "inherited_money": 0.0,
            "escheat_assets": 0,
            "escheat_money": 0.0,     # 相続人不存在で街の外(国庫)へ出た現金
            "sales": 0,
        }
        sim._ledger_state = st
    return st


def state_of(sim):
    """ON のときだけ state を返す(OFF は None = checkpoint も summary もキーを作らない)。"""
    return getattr(sim, "_ledger_state", None)


def _bump(table: dict, key: str, n: int = 1) -> None:
    table[str(key)] = int(table.get(str(key), 0)) + int(n)


# --------------------------------------------------------------------------- #
# 台帳の読み(**純関数・O(1)**。観測もシムもここしか読まない)
# --------------------------------------------------------------------------- #
def owner_of(sim, asset_id: str, right: str = OWN) -> str | None:
    """その資産の権利者(既定 = 所有者)。台帳に無ければ None。**逆引きが O(1)**。"""
    st = state_of(sim)
    if st is None:
        return None
    row = st["rows"].get(str(asset_id))
    if not row:
        return None
    got = row.get(str(right))
    return None if got is None else str(got["party"])


def assets_of(sim, party: str) -> list[str]:
    """その party が権利を持つ資産 id(**id 昇順の決定論**)。"""
    st = state_of(sim)
    if st is None:
        return []
    return sorted(st["by_owner"].get(str(party), {}))


def n_assets(sim, cat: str | None = None) -> int:
    """台帳の生存行数(カテゴリ指定可)。"""
    st = state_of(sim)
    if st is None:
        return 0
    if cat is None:
        return len(st["assets"])
    return sum(1 for rec in st["assets"].values() if rec["cat"] == str(cat))


# --------------------------------------------------------------------------- #
# 台帳の書き(**単一の書き手**。ここを通らない権利行の変更は存在しない)
# --------------------------------------------------------------------------- #
def _register(st: dict, aid: str, cat: str, party: str, step: int, rec: dict) -> None:
    """新しい資産を登記する(**初期配賦と製造・輸入だけが呼ぶ**)。"""
    st["assets"][aid] = dict(rec, cat=str(cat))
    st["rows"][aid] = {OWN: {"party": str(party), "since": int(step)}}
    st["by_owner"].setdefault(str(party), {})[aid] = OWN


def transfer(sim, asset_id: str, to_party: str, kind: str, step: int,
             right: str = OWN) -> str | None:
    """権利行の party を付け替える(**移転の唯一の口**)。戻り値 = 元の party(無ければ None)。

    ★資産保存則の意味: **行は生まれも消えもしない**(生まれるのは製造・輸入だけ・消えるのは
      廃棄・滅失だけ)。ここは「付け替え」しかしないので、生存行数は 1 も動かない。
    ★移転語彙が :data:`TRANSFER_KINDS` に無ければ ``unclassified`` へ数える
      (IF-E の未分類監視装置の双対 = 新しい移転経路が入った瞬間に鳴る)。
    """
    st = state_of(sim)
    if st is None:
        return None
    aid, dst = str(asset_id), str(to_party)
    row = st["rows"].get(aid)
    if not row or str(right) not in row:
        return None
    src = str(row[str(right)]["party"])
    if src == dst:
        return src
    holds = st["by_owner"].get(src)
    if holds is not None:
        holds.pop(aid, None)
        if not holds:
            st["by_owner"].pop(src, None)
    st["by_owner"].setdefault(dst, {})[aid] = str(right)
    row[str(right)] = {"party": dst, "since": int(step)}
    if str(kind) in TRANSFER_KINDS:
        _bump(st["transfers"], str(kind))
    else:
        _bump(st["unclassified"], str(kind))
    return src


# --------------------------------------------------------------------------- #
# 資産保存則(貨幣保存則 IF-E の完全な双対)
# --------------------------------------------------------------------------- #
def conservation(sim) -> dict:
    """カテゴリ別 Σ 検査 ``Σ(所有者別保有数) + K5 累積 − RoW 生成累積 − 境界流入 = 初期ストック``。

    戻り値 ``{cat: {"live", "born", "k5", "rot_in", "stock0", "residual"}}``。``residual`` が
    0 でないカテゴリは「無から生まれた資産」か「行方不明の資産」がある(= 台帳を通さない
    書き込みがどこかに在る)。``economy_sfc.total_money`` と同じ「閉じた不変量」の作法。

    ★``rot_in``(境界流入 = A12)は SNA 2008 の資産変動三分法のうち**境界フロー**にあたる。
      プール回転で day1 以降に街へ入ってきた個体の車両は「初期ストックの数え漏れ」ではなく
      **域外から域内へ入ってきた実在の資産**なので、``stock0`` を後から書き足すのではなく
      恒等式の流入側に立てる(そうしないと保存則の右辺が走行中に動き、検査が「あとから
      辻褄を合わせる装置」になる)。``born``(製造・輸入 = O2)とは別項に分けてあるので、
      「回転で入ってきた」のか「街の中で作られた」のかが事後に必ず区別できる。
    """
    st = state_of(sim)
    out: dict = {}
    if st is None:
        return out
    rot = st.get("rot_in") or {}                   # 旧 checkpoint 互換(キー欠落は 0 扱い)
    live: dict[str, int] = {}
    for holds in st["by_owner"].values():          # ★**所有者別**に数える(台帳の左辺そのもの)
        for aid in holds:
            rec = st["assets"].get(aid)
            if rec is not None:
                live[rec["cat"]] = live.get(rec["cat"], 0) + 1
    for cat in sorted(set(live) | set(st["stock0"]) | set(st["born"]) | set(st["k5"])
                      | set(rot)):
        n = int(live.get(cat, 0))
        born = int(st["born"].get(cat, 0))
        k5 = int(st["k5"].get(cat, 0))
        rin = int(rot.get(cat, 0))
        s0 = int(st["stock0"].get(cat, 0))
        out[cat] = {"live": n, "born": born, "k5": k5, "rot_in": rin, "stock0": s0,
                    "residual": n + k5 - born - rin - s0}
    return out


# --------------------------------------------------------------------------- #
# ① 初期配賦(**唯一の確率事象**。新 stream "asset_alloc" を 1 本だけ引く)
# --------------------------------------------------------------------------- #
def _landlord_orgs(sim, cfg: dict) -> tuple[list[tuple[str, float]], str]:
    """域内の不動産 org(家主の担い手)を組織台帳から特定する。戻り値 = ([(org_id, 重み)], 出所)。

    第 1 段(本命)= 台帳の実在フィールドだけで決まる **仮定の要らない特定**:
      ``industry_key == "RE"``(不動産業・物品賃貸業)かつ
      ``sector_detail ∈ {"賃貸仲介/管理", "開発/PM"}``。
      オフィス仲介・物品賃貸は住戸の家主ではないので外す。
    第 2 段(後退)= 該当が 1 社も無い台帳(sector_detail を持たない旧版など)では
      ``workplace_poi.cat == "office"`` の org の**決定論部分集合**(id 昇順の先頭)を
      「管理会社と見なす」。★これは**仮定**なので ``org_source`` に必ず残す。
    org 台帳そのものが無い(organizations OFF)世界では空 = 賃貸住戸は全部 RoW(不在家主)= 現行と同じ。

    重み = 従業者数(``size.employees``。最低 1)。大きい会社ほど多く持つ(均等割りにすると
    所有ネットワークが退化して集中度の観測が死ぬ)。
    """
    book = getattr(sim, "orgs", None) or {}
    if not book:
        return [], "none"
    key = str(cfg["landlord_industry_key"])
    sectors = set(cfg["landlord_sectors"])
    cands = sorted(oid for oid, org in book.items()
                   if str(org.get("industry_key") or "") == key
                   and str(org.get("sector_detail") or "") in sectors)
    source = "census_industry_key"
    if not cands:                                  # 後退: office の決定論部分集合(**仮定**)
        cands = sorted(oid for oid, org in book.items()
                       if str((org.get("workplace_poi") or {}).get("cat") or "") == "office")
        source = "office_fallback"
    if not cands:
        return [], "none"
    cands = cands[:max(1, int(cfg["max_landlord_orgs"]))]
    out: list[tuple[str, float]] = []
    for oid in cands:
        size = (book[oid].get("size") or {})
        out.append((str(oid), max(1.0, float(size.get("employees", 1) or 1))))
    return out, source


def _landlord_index(sim, cfg: dict) -> tuple[list, list, str]:
    """家主 org の一覧・従業者数の累積和・特定方法(**起動時 1 回だけ台帳を走る**)。

    ★キャッシュしないと ``_on_move`` が転居 1 件ごとに org 台帳(本番 9,872 社)を全走査する。
      キャッシュ属性 ``sim._ledger_landlords`` は台帳(state)ではない**派生**なので
      checkpoint には載せない(resume 側で同じ台帳から同じ順序で組み直る = 決定論)。
    """
    got = getattr(sim, "_ledger_landlords", None)
    if got is None:
        landlords, source = _landlord_orgs(sim, cfg)
        cum: list[float] = []
        acc = 0.0
        for _, w in landlords:
            acc += float(w)
            cum.append(acc)
        got = (landlords, cum, source)
        sim._ledger_landlords = got
    return got


def _pick_landlord(landlords: list, cum: list, u: float) -> str:
    """重み付き累積からの決定論選択(``u`` は sha256 ジッタ = 乱数を 1 粒も消費しない)。"""
    total = cum[-1]
    target = float(u) * total
    lo, hi = 0, len(cum) - 1
    while lo < hi:                                 # 二分探索(家主 org は最大 512 件)
        mid = (lo + hi) // 2
        if cum[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return landlords[lo][0]


def _occupancy(sim) -> dict:
    """(住宅建物, 階) → **代表居住者の agent id**(最小 id)。1 step 1 回の走査だけ。

    代表を最小 id にするのは ``household._assign_household``(世帯の代表 = 最小 id)と同じ規約。
    来街者(街の外に家がある)と死者は住戸を占めない。
    """
    out: dict[tuple[str, int], int] = {}
    for a in sim.agents:                           # sim.agents 由来 = id 昇順 = 決定論
        if bool(getattr(a, "visitor", False)) or bool(getattr(a, "dead", False)):
            continue
        bld = str(getattr(a, "home_building", "") or "")
        if not bld:
            continue
        key = (bld, int(getattr(a, "home_floor", 1) or 1))
        prev = out.get(key)
        if prev is None or int(a.id) < prev:
            out[key] = int(a.id)
    return out


def _alloc_dwellings(sim, cfg: dict, st: dict, rng, step: int) -> None:
    """住戸の初期配賦(**1 戸 1 draw**)。所有者 = 居住者(持ち家)/ 域内不動産 org / RoW。

    ユーザー決定 §5-1「域内の不動産会社も所有者に追加する」の実装:
      - 居住者が居る住戸 … ``owner_occupancy_rate`` の確率で**その居住者の持ち家**、
        残りは賃貸 → ``org_landlord_share`` で域内 org / それ以外は RoW(域外家主)
      - 空き住戸 … 賃貸と同じ分岐(空き家も誰かの持ち物である)
    どの org が持つかは**従業者数の重み付き決定論**(sha256 ジッタ = 乱数消費ゼロ)。
    """
    city = getattr(sim, "city", None)
    blds = sorted((getattr(city, "residential_buildings", None) or []),
                  key=lambda b: str(b["id"]))
    if not blds:
        return
    landlords, cum, source = _landlord_index(sim, cfg)
    st["org_source"] = source
    st["landlords"] = [oid for oid, _ in landlords]
    seed = _seed_of(sim)
    occ = _occupancy(sim)
    own_rate = float(cfg["owner_occupancy_rate"])
    org_share = float(cfg["org_landlord_share"]) if landlords else 0.0
    cap = int(cfg["max_assets"])
    for b in blds:
        bid = str(b["id"])
        levels = max(1, int(b.get("levels", 1) or 1))
        for floor in range(1, levels + 1):
            if len(st["assets"]) >= cap:
                st["capped"] = True
                return
            aid = dwelling_id(bid, floor)
            resident = occ.get((bid, floor))
            u = float(rng.random())                # ★唯一の確率事象(1 戸 1 draw)
            if resident is not None and u < own_rate:
                party, label = agent_party(resident), ALLOC_RESIDENT
            else:
                # 賃貸(または空き家)= 家主が持つ。域内 org / 域外(RoW)の 2 択。
                v = (u - own_rate) / (1.0 - own_rate) if (
                    resident is not None and own_rate < 1.0) else u
                if landlords and v < org_share:
                    party = org_party(_pick_landlord(
                        landlords, cum, _jitter(seed, "landlord", aid)))
                    label = ALLOC_ORG
                else:
                    party, label = row_party(), ALLOC_ROW
            cx, cy = (b.get("centroid") or (0.0, 0.0))[:2]
            _register(st, aid, DWELLING, party, step,
                      {"building": bid, "floor": int(floor),
                       "node": str(b.get("entrance") or ""),
                       "x": float(cx), "y": float(cy),
                       "occupied": bool(resident is not None)})
            _bump(st["alloc"], label)


def _alloc_vehicles(sim, cfg: dict, st: dict, step: int, *, inflow: bool = False) -> int:
    """車両の登記 = ``has_car`` の**個体昇格**(bool タグ → 台帳の行)。乱数ゼロ・冪等。

    ``has_car`` は既にペルソナ生成時に決まっている(= 抽選済み)ので、ここで引き直すと
    同じ事実を 2 回抽選することになる。所有者は必ず本人(車の所有者が世界に居ないという
    穴は最初から無い)。

    ``inflow=False``(:func:`arm` からの初期配賦)= 登記した行はそのまま ``stock0`` に入る。
    ``inflow=True``(:func:`_follow_vehicles` からの日次追随)= **境界流入**として
    ``rot_in`` に計上する(:func:`conservation` の docstring)。戻り値 = 登記した行数。
    """
    cap = int(cfg["max_assets"])
    n = 0
    for a in sim.agents:                           # id 昇順 = 決定論
        if not bool(getattr(a, "has_car", False)) or bool(getattr(a, "dead", False)):
            continue
        aid = vehicle_id(a.id)
        if aid in st["assets"]:
            continue                               # 既に登記済み(再来街・二度目の走査)= 冪等
        if len(st["assets"]) >= cap:
            st["capped"] = True
            return n
        _register(st, aid, VEHICLE, agent_party(a.id), step,
                  {"building": "", "floor": 0,
                   "node": str(getattr(a, "home_node", "") or ""),
                   "x": float(getattr(a, "x", 0.0)), "y": float(getattr(a, "y", 0.0)),
                   "agent": int(a.id)})
        if inflow:
            _bump(st.setdefault("rot_in", {}), VEHICLE)
        n += 1
    return n


def _follow_vehicles(sim, cfg: dict, st: dict, step: int, sim_min: int) -> int:
    """A12 車両資産の**日次追随**(日境界に 1 回・乱数ゼロ・冪等・登記済みはスキップ)。

    なぜ要るか(現物の穴): :func:`_alloc_vehicles` は :func:`arm` からしか呼ばれず、
    登記されるのは **day0 の在場者の車だけ**である。プール回転で day1 以降に入場する個体
    (finals 構成で 20.9 万人)の ``has_car`` は永久に bool タグのままで、台帳にも
    ``holder_gini`` にも所有ネットワークにも 1 行も現れない = 資産分布が day0 在場者という
    「たまたま初日に街に居た人」の標本に痩せていた。

    なぜ ``stock0`` に足さないか: ``stock0`` は保存則の**右辺**(初期ストック)である。
    後から登記した行をそこへ入れると右辺が走行中に動き、Σ 残差 0 は「常に辻褄が合う」
    という無内容な恒等式になる。ここでは流入側の独立項 ``rot_in`` に計上するので、
    保存則は ``live + k5 − born − rot_in = stock0`` の強い形のまま生き残る
    (``city_ops._ensure_bound`` の日次追随と同型 = 枠方式・乱数ゼロ・L1 増ゼロ)。

    既定プロファイル(pool OFF)では day1 以降に新規入場が無いので、走査しても登記は
    1 行も増えない = **バイト不変**。
    """
    if not cfg["vehicles"] or not st["init"]:
        return 0
    day = int(sim_min) // 1440
    if int(st.get("veh_day", 0)) == day:
        return 0                                   # 同一日に 2 度走らない(冪等ガード)
    st["veh_day"] = day
    return _alloc_vehicles(sim, cfg, st, int(step), inflow=True)


def arm(sim, step: int, sim_min: int) -> None:
    """起動時 1 回の初期配賦(**既定 OFF は即 return**)。resume では checkpoint 復元済み = 走らない。

    ``economy_sfc._sfc_arm``(org 預金の期首配賦)と同じ位置・同じ規約: 乱数は新 stream
    ``"asset_alloc"`` 1 本だけ・L1 を 1 件も出さない・世界状態を 1 バイトも変えない。
    """
    if not enabled(sim):
        return
    st = _state(sim)
    if st["init"]:
        return
    cfg = cfg_of(sim)
    rng = sim.hub.stream("asset_alloc")             # ★新 stream 1 本(既存 draw 順に不干渉)
    if cfg["dwellings"]:
        _alloc_dwellings(sim, cfg, st, rng, int(step))
    if cfg["vehicles"]:
        _alloc_vehicles(sim, cfg, st, int(step))
    for aid, rec in st["assets"].items():           # 初期ストック = 保存則の右辺
        _bump(st["stock0"], rec["cat"])
    st["init"] = True


# --------------------------------------------------------------------------- #
# ② 既存イベントの台帳写像(**挙動 1 バイト不変 = 分類の追加だけ**)
#
#   新しい L1 の watermark 走査(``rumors._new_events`` と同型)。死亡・転居があった step
#   だけ仕事をするので、毎 step の全資産走査はゼロ。
#   watermark は**プロセス内 logger カウンタ由来**なので checkpoint に保存しない
#   (resume 直後は新しい logger の 0 から数え直す = 第59 assets / 第61 gossip と同流儀)。
#   ★二重処理は起きない: 相続は故人 id の集合(``heired``)で冪等、売却は「所有と居住が
#     食い違う住戸」という**状態**を見るので、同じ状態を 2 度直すことはない。
# --------------------------------------------------------------------------- #
_WATCH_KINDS: frozenset[str] = frozenset({"death", "move_home", "relocate"})


def _new_events(sim) -> list:
    logger = getattr(sim, "logger", None)
    if logger is None:
        return []
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)
    processed = int(getattr(sim, "_ledger_watermark", 0))
    new = max(0, min(total - processed, len(events)))
    sim._ledger_watermark = total
    return events[len(events) - new:] if new else []


def _heirs(sim, agent) -> tuple[list, int]:
    """相続人 = **同一世帯の生存者**(``household_id`` の受け皿 = ``housemates``)。id 昇順。

    戻り値 = (相続人, **手が届かなかった世帯員の数**)。

    ★``household.enabled=false`` の世界では ``housemates`` が空なので相続人は 0 人になる
      (= 世帯という受け皿が世界に無い)。そのとき遺産は国庫へ出る(:func:`_escheat`)。
      「世帯が無いから相続が起きない」ことを隠さないのが本実装の立場。
    ★プール回転で**いま街に居ない**世帯員は相続人に数えられない(退場中の個体は脱水されて
      いて残高を書けない)。黙って落とすと「配偶者が居るのに国庫へ行った」が観測から消えるので、
      その人数を第 2 戻り値として返し L1 の ``absent`` に載せる。
    ★★レーン甲(2026-08-13)の訂正: 以前ここは ``sim.agent_by_id.get(hid) is None`` を
      「街に居ない」の判定に使っていたが、``agent_by_id`` は**これまで実体化した全個体**の索引で
      退場者を消さない = **None は決して返らない**。つまり不在の相続人へ ``money +=`` を書いて
      いて、その書き込みは次の hydrate(退場時スナップショット)で捨てられ、故人の残高だけが
      0 になっていた = **現金が世界から消える**。判定は ``sim.present_agent``(在場述語)で行う。
    """
    out = []
    absent = 0
    mates = sorted({int(i) for i in (getattr(agent, "housemates", None) or ())
                    if int(i) != int(agent.id)})
    for hid in mates:
        h = sim.present_agent(hid)
        if h is None:                              # プール回転で街に居ない = 手が届かない
            absent += 1
            continue
        if bool(getattr(h, "dead", False)):        # 故人は相続人にならない(absent とは別)
            continue
        out.append(h)
    return out, absent


def _cash_of(agent) -> tuple[float, float]:
    return (max(0.0, float(getattr(agent, "money", 0.0) or 0.0)),
            max(0.0, float(getattr(agent, "account", 0.0) or 0.0)))


def _split(total: float, n: int) -> list[float]:
    """n 人への等分(**合計は 1 円もずれない**: 最後の 1 人が端数を引き取る)。"""
    if n <= 0 or total <= 0.0:
        return []
    each = round(float(total) / n, 6)
    out = [each] * (n - 1)
    out.append(round(float(total) - each * (n - 1), 6))
    return out


def _inherit_money(agent, heirs: list) -> float:
    """現金と口座残高を相続人へ等分する(**街の総額は 1 円も動かない**)。戻り値 = 動いた額。"""
    money, account = _cash_of(agent)
    moved = 0.0
    for field, total in (("money", money), ("account", account)):
        if total <= 0.0:
            continue
        for h, amt in zip(heirs, _split(total, len(heirs))):
            setattr(h, field, round(float(getattr(h, field, 0.0) or 0.0) + amt, 6))
            moved += amt
        setattr(agent, field, 0.0)
    return round(moved, 6)


def _escheat(sim, st: dict, agent) -> float:
    """相続人不存在 → **国庫へ帰属**(民法 959 条)。国は街の外なので RoW チャネルへ落とす。

    ★``lost_property`` の hold バケツと同じ作法: 本 module が**自分のバケツ**
      (``escheat_money``)で金の行き先を持ち、IF-E2 が ON のときだけ RoW にも記帳する
      (OFF のランで ``economy_sfc`` の state を生やさない = 相手の R1 を壊さない)。
    """
    money, account = _cash_of(agent)
    total = round(money + account, 6)
    if total <= 0.0:
        return 0.0
    agent.money = 0.0
    if account > 0.0:
        agent.account = 0.0
    st["escheat_money"] = round(float(st["escheat_money"]) + total, 6)
    sfc_mod.on_inheritance_escheat(sim, total)
    return total


def _on_death(sim, cfg: dict, st: dict, agent, step: int, sim_min: int) -> None:
    """★相続(O3): 故人の**現金 + 資産行**を世帯へ移す。L1 は **1 行だけ**。

    第107 の観測盲点(死者の財布に凍結)の解消そのもの。受け皿が居ない回は国庫(RoW)へ
    出し、``to="row"`` として**正直に名付ける**(黙って凍らせない)。
    """
    heirs, absent = _heirs(sim, agent)
    owned = assets_of(sim, agent_party(agent.id))
    st["deaths"] = int(st["deaths"]) + 1
    st["heirs_absent"] = int(st.get("heirs_absent", 0)) + int(absent)
    if heirs:
        moved = _inherit_money(agent, heirs)
        for k, aid in enumerate(owned):            # id 昇順 → 相続人へ順に(決定論)
            transfer(sim, aid, agent_party(heirs[k % len(heirs)].id), INHERIT, step)
        st["inherited_money"] = round(float(st["inherited_money"]) + moved, 6)
        st["inherited_assets"] = int(st["inherited_assets"]) + len(owned)
        to = "household"
    else:
        moved = _escheat(sim, st, agent)
        for aid in owned:
            transfer(sim, aid, row_party(), ESCHEAT, step)
        st["escheat_assets"] = int(st["escheat_assets"]) + len(owned)
        to = "row"
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=int(agent.id),
                         kind="inheritance", x=float(agent.x), y=float(agent.y),
                         payload={"heirs": len(heirs), "assets": len(owned),
                                  "amount": round(moved, 1), "to": to,
                                  # ★街に居ない世帯員(プール退場中)= 手が届かなかった相続人。
                                  #   0 でも載せる(「居なかった」と「見なかった」を分ける)
                                  "absent": int(absent)}))


def _on_move(sim, cfg: dict, st: dict, agent, step: int, sim_min: int) -> int:
    """持ち家者の転居 = **``own`` の移転**(買い手 = 域内 org / RoW の決定論)。戻り値 = 出した L1 件数。

    「自分が所有していて、もう自分が住んでいない住戸」を売却として台帳に写す。
    ★**金は 1 円も動かさない**(module docstring の正直な限界 1)。売買代金の内生化は O5。
    ★賃借人の転居は台帳を 1 行も動かさない(``own`` は家主のままだから)= 分類として正しい。
    """
    st_assets = st["assets"]
    bld = str(getattr(agent, "home_building", "") or "")
    floor = int(getattr(agent, "home_floor", 1) or 1)
    landlords, cum, _src = _landlord_index(sim, cfg)
    seed = _seed_of(sim)
    n = 0
    for aid in assets_of(sim, agent_party(agent.id)):
        rec = st_assets.get(aid)
        if rec is None or rec["cat"] != DWELLING:
            continue
        if str(rec["building"]) == bld and int(rec["floor"]) == floor:
            continue                               # まだそこに住んでいる = 何も起きない
        if landlords and _jitter(seed, "buyer", aid, int(step)) < float(
                cfg["org_landlord_share"]):
            buyer = org_party(_pick_landlord(landlords, cum,
                                             _jitter(seed, "landlord", aid)))
        else:
            buyer = row_party()
        src = transfer(sim, aid, buyer, SALE, int(step))
        st["sales"] = int(st["sales"]) + 1
        n += 1
        sim.logger.log(Event(step=int(step), sim_min=int(sim_min),
                             agent_id=int(agent.id), kind="asset_transfer",
                             x=float(agent.x), y=float(agent.y),
                             payload={"asset": aid, "cat": DWELLING, "right": OWN,
                                      "from": str(src or ""), "to": buyer,
                                      "kind": SALE}))
    return n


def phase(sim, step: int, sim_min: int) -> None:
    """既存イベントの台帳写像(**既定 OFF は即 return**)。

    ★走るのは「新しい L1 に死亡 / 転居があった step」だけ = 毎 step の全資産走査はゼロ。
    ★世界状態のうち本 module が動かすのは **現金(相続)だけ**(位置・関係・drive・opinion・
      記憶・プロンプトは 1 つも触らない)。LLM を 1 度も呼ばない。
    """
    if not enabled(sim):
        return
    st = _state(sim)
    cfg = cfg_of(sim)
    # A12: 車両資産の日次追随(日境界の 1 回だけ・登記済みはスキップ・L1 を 1 件も出さない)。
    # ★``_new_events`` の**前**に置く: watermark は「新しい L1 を 1 度だけ見る」ための
    #   カウンタなので、途中 return する経路の後ろに置くと日次追随が飛ぶ日が出る。
    _follow_vehicles(sim, cfg, st, int(step), int(sim_min))
    events = _new_events(sim)
    if not events:
        return
    budget = int(cfg["max_events_per_step"])
    heired = set(st["heired"])
    for e in events:
        if budget <= 0:
            break
        if e.kind not in _WATCH_KINDS:
            continue
        # ★当事者(故人 / 転居者)も**在場者に限る**。退場者の幽霊で相続を回すと、故人側の
        #   ``money = 0`` が hydrate で捨てられる一方で相続人には着金する = **現金が湧く**。
        #   居ない回は何もしない(遺産は退避 record に凍ったまま = 総額は保存される)。
        agent = sim.present_agent(int(e.agent_id))
        if agent is None:
            continue
        if e.kind == "death":
            if int(agent.id) in heired:
                continue                           # 冪等(同じ死を 2 度相続しない)
            heired.add(int(agent.id))
            _on_death(sim, cfg, st, agent, int(step), int(sim_min))
            budget -= 1
        else:                                      # move_home / relocate
            budget -= _on_move(sim, cfg, st, agent, int(step), int(sim_min))
    st["heired"] = sorted(heired)


# --------------------------------------------------------------------------- #
# ③ 観測(summary.json の "assets" キー。**OFF ではキー自体を出さない**)
# --------------------------------------------------------------------------- #
def _gini(counts: list) -> float:
    """保有数の Gini(0=平等 .. 1=集中)。``observer/aggregate._gini`` と同式の自前実装。"""
    vals = sorted(float(v) for v in counts)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total <= 0.0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return round((2.0 * cum) / (n * total) - (n + 1.0) / n, 6)


def holdings(sim, cat: str | None = None) -> dict:
    """party → 保有数(``cat`` 指定でカテゴリ別)。**台帳の純関数**(シムに走査を足さない)。"""
    st = state_of(sim)
    out: dict[str, int] = {}
    if st is None:
        return out
    for party, holds in st["by_owner"].items():
        n = 0
        for aid in holds:
            rec = st["assets"].get(aid)
            if rec is not None and (cat is None or rec["cat"] == str(cat)):
                n += 1
        if n:
            out[str(party)] = n
    return out


def provenance(sim) -> dict | None:
    """``summary.json`` の ``assets`` キー(既定 OFF は None = キー自体を出さない)。

    ★**資産保存則の残差**(``conservation``)と**未分類の移転種**(``unclassified``)を
      一級市民として出す = 「無から生まれた資産ゼロ・行方不明の資産ゼロ」を事後に機械検証できる。
    ★初期配賦の内訳(持ち家 / 域内 org / 域外 RoW)と**域内不動産 org の特定方法**
      (``org_source``)も残す = 較正のどこが仮定なのかが 1 枚で判る。
    """
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    out: dict = {"schema": SCHEMA,
                 "categories": list(CATEGORIES),
                 "right_kinds": list(cfg["right_kinds"]),
                 "owner_occupancy_rate": float(cfg["owner_occupancy_rate"]),
                 "org_landlord_share": float(cfg["org_landlord_share"])}
    st = state_of(sim)
    if st is None:                                 # ON だが arm 前(1 行も登記していない)
        out.update({"n_assets": 0, "stock0": {}, "rot_in": {}, "alloc": {},
                    "conservation": {},
                    "transfers": {}, "unclassified": {}, "org_source": "none",
                    "n_landlord_orgs": 0, "by_party_kind": {}, "holder_gini": 0.0,
                    "deaths": 0, "heirs_absent": 0,
                    "inherited_assets": 0, "inherited_money": 0.0,
                    "escheat_assets": 0, "escheat_money": 0.0, "sales": 0,
                    "capped": False})
        return out
    hold = holdings(sim)
    by_kind: dict[str, int] = {}
    for party, n in hold.items():
        by_kind[party_kind(party)] = by_kind.get(party_kind(party), 0) + int(n)
    out.update({
        "n_assets": len(st["assets"]),
        "stock0": {k: int(v) for k, v in sorted(st["stock0"].items())},
        # ★境界流入(A12)= プール回転で街に入ってきた既存資産の累積。0 でないランは
        #   「初期ストックだけでは資産分布が痩せていた」ことの実測そのもの。
        "rot_in": {k: int(v) for k, v in sorted((st.get("rot_in") or {}).items())},
        "alloc": {k: int(v) for k, v in sorted(st["alloc"].items())},
        "conservation": conservation(sim),
        "transfers": {k: int(v) for k, v in sorted(st["transfers"].items())},
        # ★未分類の移転種(IF-E の監視装置の双対。空であることをテストが固定する)
        "unclassified": {k: int(v) for k, v in sorted(st["unclassified"].items())},
        "org_source": str(st["org_source"]),
        "n_landlord_orgs": len(st["landlords"]),
        "n_holders": len(hold),
        "by_party_kind": {k: int(v) for k, v in sorted(by_kind.items())},
        # 所有の集中(agent / org / RoW を party として同列に数えた保有数の Gini)
        "holder_gini": _gini(list(hold.values())),
        "deaths": int(st["deaths"]),
        # ★街に居なかった世帯員の累計。ここが大きいランは「配偶者が居るのに国庫へ行った」が
        #   多い = プール回転の在場率が相続の受け皿を痩せさせている(正直な限界 7)
        "heirs_absent": int(st.get("heirs_absent", 0)),
        "inherited_assets": int(st["inherited_assets"]),
        "inherited_money": round(float(st["inherited_money"]), 1),
        "escheat_assets": int(st["escheat_assets"]),
        "escheat_money": round(float(st["escheat_money"]), 1),
        "sales": int(st["sales"]),
        "capped": bool(st["capped"]),
    })
    return out


def cash_escheated(sim) -> float:
    """相続人不存在で街から出た現金の累計(**読むだけ**・state も属性も生やさない)。

    貨幣保存の検査に使う項: ``Σ agent.money(+account) + escheat 累計`` が一定。
    """
    st = state_of(sim)
    return 0.0 if st is None else float(st["escheat_money"])


# --------------------------------------------------------------------------- #
# ④ 台帳サイドカー(``assets_ledger.json``)
#
#   解析(``scripts/analyze_assets.py``)が Gini・保有期間・所有ネットワークを計算する材料。
#   **シム本体に走査を足さない**ため、書き出しは finalize の 1 回だけ(``truth_ledger.dump``
#   と同じ位置・同じ流儀)。OFF のランでは 1 ファイルも作らない。
# --------------------------------------------------------------------------- #
def dump(sim) -> None:
    """登記簿のスナップショットを ``assets_ledger.json`` へ書き出す(OFF は no-op)。"""
    if not enabled(sim):
        return
    st = state_of(sim)
    if st is None:
        return
    import json
    rows = []
    for aid in sorted(st["rows"]):
        rec = st["assets"].get(aid) or {}
        for right in sorted(st["rows"][aid]):
            r = st["rows"][aid][right]
            rows.append({"asset": aid, "cat": str(rec.get("cat", "")),
                         "right": str(right), "party": str(r["party"]),
                         "since": int(r["since"]),
                         "building": str(rec.get("building", "")),
                         "floor": int(rec.get("floor", 0)),
                         "node": str(rec.get("node", ""))})
    payload = {"schema": SCHEMA, "n_rows": len(rows),
               "stock0": {k: int(v) for k, v in sorted(st["stock0"].items())},
               # ★保存則の 3 つの帳尻項を**サイドカーにも**載せる: 解析側
               #   (scripts/analyze_assets.py)がシム側の申告を見ずに残差を数え直せるように
               #   するため。これが無いと A12(境界流入)のあるランで解析だけが FAIL になる。
               "born": {k: int(v) for k, v in sorted(st["born"].items())},
               "k5": {k: int(v) for k, v in sorted(st["k5"].items())},
               "rot_in": {k: int(v) for k, v in sorted((st.get("rot_in") or {}).items())},
               "conservation": conservation(sim),
               "landlords": list(st["landlords"]),
               "org_source": str(st["org_source"]),
               "rows": rows}
    (sim.out_dir / "assets_ledger.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
