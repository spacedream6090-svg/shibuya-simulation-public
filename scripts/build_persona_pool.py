"""ペルソナ100万プールの決定論生成(W2 P5 / batch: persona-pool 1M)。

使い方:
    python scripts/build_persona_pool.py                       # フル(~100万)
    python scripts/build_persona_pool.py --fraction 0.02       # 小規模(テスト/検証用)
    python scripts/build_persona_pool.py --seed 42 --out data/persona_pool

設計(docs/plans/persona-pool.md §2/§5 + docs/plans/w2-execution-plan.md §4 P5):
  5層をすべて**決定論**(seed 固定・LLM 不使用)で生成し、シャーディングした JSONL に吐く。
    L1 住民      : shibuya_population.json の周辺分布から IPF で骨格(夜間人口 ~3万)
    L2 域内従業者: 組織台帳(既定 organizations_shibuya_wide11k.json・--orgs で差替可)の employees を需要源に逆算
                   (会社 employees の総和 + 学校の教職員)。org_id/role/shift を本人に埋め込む
    L3 定期来街  : 学生(学校 capacity から逆算・org_id=学校)+ 常連(習い事等・週次)
    L4 非定期来街: 回転の主層(観光/買物/ビジネス来訪の匿名合成セグメント。数十万)
    L5 役割ペルソナ: 駅員/電車運転士/バス運転士/タクシー運転手/警察官/配信者 + 議員
  各レコードに layer タグと presence(§5 のローテーション区分)を付与する。

決定論:
  乱数はパートごとに numpy の SeedSequence([master_seed, layer_code, part_index]) から導出。
  → 同 seed・同 fraction なら**他パートの実行順に依存せず**同一出力(シャード並列可能)。

スキーマ互換:
  フィールド名は data/personas_300_civic.json に揃える(name/age/gender/occupation/visitor/
  commute/persona/traits/drive_threshold/fire_weight/bedtime_min/sleep_steps/has_bicycle/
  has_car (+commuter: arrival_lead_min/commute_gateway/residence_line))。
  追加フィールド(id/layer/presence/org_id/role/shift_pattern/visit_cadence/visit_purpose/
  visit_rate/is_foreign/party_size/revisit/post/duty_pattern/seat_id/party)はスキーマ拡張。
  拡張分は persona.build_agent が entry.get で無視するため後方互換(engine 非改変)。

出力(data/persona_pool/):
  {layer}/part-XXXX.jsonl(1行=1レコード)+ meta.json(層別件数・seed・スキーマ版・シャード一覧)
  + llm_targets.json(深いペルソナ化=LLM上塗り対象の id 一覧: L5 全員 + L1/L3常連の一部)。
  ※ data/persona_pool/ は .gitignore 済み(数百MB級はコミットしない)。

副産物(コミット対象・小):
  data/personas_councilors.json … 議員名簿(標準スキーマ・from_roster で着席可能)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

try:                                       # cp932 コンソールでの print 死を回避
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):       # pytest capture 等では reconfigure 不可 → 無視
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "persona-pool-1.0"
PART_SIZE = 50_000            # 1 シャードの最大レコード数

# 組織台帳の既定パス(従来のハードコードをそのまま既定値へ移しただけ = 無指定なら 1 バイトも
# 変わらない)。--orgs でセンサス較正台帳 data/organizations_shibuya_census.json 等に差し替える。
# ★docs/research/economy-census-calibration.md §8 ④ が既にこの引数を前提に書かれていた
#   (「引数は同スクリプトの --help を参照」)が、実装が無かった = 本選前リビルドの詰まり。
DEFAULT_ORGS_FILE = "data/organizations_shibuya_wide11k.json"

# ---- 層コード(SeedSequence の第2要素。決定論の名前空間分離)----
LC = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}

# ---- 名前(手続き生成・実在人物を想起させない一般的な姓名の組合せ)----
_FAMILY = [
    "佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
    "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "斎藤", "清水",
    "山崎", "森", "池田", "橋本", "阿部", "石川", "山下", "中島", "石井", "小川",
    "前田", "岡田", "長谷川", "藤田", "後藤", "近藤", "村上", "遠藤", "青木", "坂本",
    "福田", "太田", "西村", "藤井", "岡本", "三浦", "松田", "中川", "中野", "原田",
    "小野", "田村", "竹内", "金子", "和田", "中山", "石田", "上田", "森田", "宮本",
]
_GIVEN_F = [
    "美咲", "陽菜", "結衣", "葵", "凛", "さくら", "芽衣", "千夏", "真央", "花音",
    "美月", "詩織", "彩乃", "智子", "由美", "恵", "直美", "幸子", "京子", "和子",
    "愛", "あや", "莉子", "杏", "楓", "七海", "美優", "彩香", "沙織", "麻衣",
    "奈々", "遥", "萌", "香織", "成美", "有紗", "咲良", "美穂", "文香", "早紀",
    "瞳", "菜々子", "琴音", "優花", "千香", "真由", "千尋", "美嘉",
]
_GIVEN_M = [
    "翔太", "蓮", "大輝", "颯太", "拓海", "悠人", "健太", "亮", "光", "駿",
    "隆", "誠", "浩二", "健一", "修", "茂", "勝", "昇", "清", "実",
    "翔", "大和", "海斗", "陸", "湊", "樹", "悠斗", "航", "翼", "拓也",
    "圭", "涼介", "雄大", "和也", "直樹", "達也", "亮太", "祐介", "剛", "隼人",
    "颯", "遼", "純", "慎一", "宏", "克也", "秀樹", "洋平",
]
_FAMILY_SET = set(_FAMILY)
_GIVEN_SET = set(_GIVEN_F) | set(_GIVEN_M)

# ---- 語彙(persona 文の flavor。心理学用語は使わない)----
_HOBBIES = ["音楽ライブ", "カフェ巡り", "写真", "ランニング", "古着屋めぐり", "読書",
            "ゲーム", "自炊", "街歩き", "アニメ", "映画", "筋トレ", "サウナ", "釣り",
            "登山", "ボードゲーム", "陶芸", "DIY", "自転車", "韓国ドラマ", "旅行", "散歩"]
_TONES = ["よく笑う", "淡々と話す", "早口でよくしゃべる", "口数は少なめ", "冗談が多い",
          "丁寧な物腰", "熱く語りがち", "マイペース", "人懐っこい", "皮肉っぽい",
          "おっとりしている", "さっぱりした性格"]

_RESIDENCE_LINES = ["世田谷方面", "二子玉川・田園都市線沿線", "横浜・東横線沿線",
                    "下北沢・井の頭線沿線", "新宿・山手線沿線", "杉並方面",
                    "目黒方面", "川崎・多摩方面"]

# =========================================================================== #
# 職業名対応表(PRES-B。第109バッチ レーン PRES-Pool)
# =========================================================================== #
# 問題(計画書 §0-4): L2 域内従業者の `occupation` は**台帳のロール名そのまま**だった。
# ロール名は「社の中でどの機能を担うか」(販売スタッフ / 医療スタッフ / オペレーション /
# スタイリスト …)であって**職業名ではない**。台帳は industry_key と sector_detail を実在の
# 産業分類で持っているのに、生成器がそれを読んでいなかった。
#
# ここで (industry_key, sector_detail) × ロール → **現実の職業名**へ写像する。
#
# 規律:
#  ① **一般名詞のみ**(実在ブランド・実在企業・実在人物を想起させる語を 1 つも使わない)。
#  ② 台帳に**無い業種の職業は作らない**。センサス較正台帳 v8 には宿泊業・保育所・理容室・
#     自動車整備業の org が 1 社も無いので、「ホテルフロント」「保育士」「理容師」「整備士」は
#     **名簿に生えない**(計画の例示にあっても、居ない業種の職業を発明しない)。
#     → 生やしたければ台帳側(scripts/build_orgs.py)に業種を足すのが筋。
#  ③ **オフィス系(IT)の現行 5 種は残す**(エンジニア/デザイナー/プロダクトマネージャー/
#     営業/コーポレート)。情報通信業の実在の職種名なので置き換える理由が無い。
#  ④ **夜勤スロットのロールは触らない**(下の `_night_occupation` を参照)。夜勤ロール名は
#     conf(city_ops.night_cleaning.occupations / incidents_env.crowd.guard_occupations /
#     world.facilities.responder_occupations)が名指しで引いている語なので、ここで書き換えると
#     設定と名簿の対応が黙って切れる。
#  ⑤ 写像に無い (業種, ロール) は**ロール名のまま**(推測で埋めない)。
#
# ★挙動への影響(正直な申告): 新しい職業名のうち
#   「美容師」「カフェ店員」「アパレル店員」は persona._WORK_CAT / economy.WAGE_CAT に**既に
#   在る語**なので、その個体は職場 POI と本業日給を持つようになる(それ以外の新職業名は
#   従来どおり両表に無く日給 0 のまま)。L2 の賃金結線そのものは本レーンの範囲外
#   (= 台帳 wage_tier を個体の日給へ繋ぐかはユーザー判断)なので、この非対称は
#   **検収報告に実数つきで残す**。
#   「警備員」「設備保守員」は逆に、conf が名指ししているのに**名簿に 1 人も居なかった**語
#   (第108/109 の監査事実)で、警備業・施設管理業の実在従業者がここで初めて生える。

#: 業種横断の既定(sector_detail を見ない層)。キー = industry_key。
_OCC_BY_INDUSTRY: dict[str, dict[str, str]] = {
    # 情報通信業 = 現行 5 種を維持(規律 ③)。
    "IT": {},
    # 卸売業・小売業
    "WR": {"販売スタッフ": "販売員", "バイヤー": "仕入担当",
           "EC運営": "通信販売運営担当", "VMD": "売場装飾担当",
           "在庫管理": "在庫管理担当"},
    # 飲食業・宿泊サービス業(台帳の実体は飲食のみ)
    "FB": {"キッチン": "調理師", "ホール": "接客係"},
    # 生活関連サービス業
    "LS": {"スタイリスト": "施術スタッフ", "アシスタント": "施術アシスタント"},
    # 医療・福祉
    "MW": {"医療スタッフ": "看護師", "介護スタッフ": "介護士",
           "受付": "医療事務", "事務": "医療事務"},
    # 教育・学習支援業
    "ED": {"事務": "事務員"},
    # 娯楽業
    "AM": {"フロア": "接客係", "音響": "音響技術者"},
    # 学術研究・専門・技術サービス業(職種名が既に実在なので最小限)
    "PS": {},
    # 不動産業・物品賃貸業
    "RE": {"営業": "不動産営業", "物件管理": "不動産管理員", "事務": "不動産事務"},
    # 建設業
    "CN": {"施工管理": "施工管理技士", "設計": "建築設計士", "事務": "事務員"},
    # 金融業・保険業
    "FI": {"コンサルタント": "金融コンサルタント", "営業": "金融営業", "事務": "金融事務"},
    # 製造業
    "MF": {"企画": "商品企画担当", "生産管理": "生産管理担当", "事務": "事務員"},
    # サービス業(他に分類されないもの)
    "SV": {"オペレーション": "オペレーター", "スタッフ": "サービススタッフ",
           "事務": "事務員"},
    # 運輸業・郵便業
    "TR": {"ドライバー": "トラック運転手", "倉庫管理": "倉庫作業員",
           "オペレーション": "配車オペレーター", "事務": "事務員"},
    # 複合サービス事業
    "CS": {"窓口": "窓口係", "スタッフ": "窓口スタッフ", "事務": "事務員"},
}

#: 業種 × 細分類(sector_detail)で上書きする層。キー = (industry_key, sector_detail)。
_OCC_BY_SECTOR: dict[tuple[str, str], dict[str, str]] = {
    # ---- 小売(WR)。売る物で店員の呼び名が変わる ----
    ("WR", "アパレル小売"): {"販売スタッフ": "アパレル店員"},
    ("WR", "セレクトショップ"): {"販売スタッフ": "衣料品販売員"},
    ("WR", "コスメ物販"): {"販売スタッフ": "美容部員"},
    ("WR", "食品小売"): {"販売スタッフ": "食品販売員"},
    ("WR", "雑貨・ライフスタイル"): {"販売スタッフ": "雑貨販売員"},
    ("WR", "生活雑貨卸"): {"販売スタッフ": "卸売営業担当", "バイヤー": "仕入担当"},
    # ---- 飲食(FB)----
    ("FB", "カフェ"): {"キッチン": "調理補助", "ホール": "カフェ店員"},
    ("FB", "ベーカリー"): {"キッチン": "パン職人", "バリスタ": "製造補助",
                           "ホール": "販売員"},
    ("FB", "バー"): {"バリスタ": "バーテンダー"},
    ("FB", "テイクアウト"): {"ホール": "販売員", "バリスタ": "調理補助"},
    # ---- 生活関連サービス(LS)。ここが最も「職業名でないロール名」が目立っていた層 ----
    ("LS", "美容室"): {"スタイリスト": "美容師", "アシスタント": "美容アシスタント"},
    ("LS", "ネイル/エステ"): {"スタイリスト": "エステティシャン"},
    ("LS", "リラクゼーション"): {"スタイリスト": "セラピスト"},
    # ★「写真家」にはしない: economy.WAGE_CAT が「写真家 = 自営」と宣言していて、
    #   スタジオの被用者に出来高(gig)収入が付いてしまう。雇われの撮影者は「カメラマン」。
    ("LS", "写真スタジオ"): {"スタイリスト": "カメラマン", "アシスタント": "撮影アシスタント"},
    ("LS", "クリーニング"): {"スタイリスト": "クリーニング師", "アシスタント": "仕上げ担当"},
    ("LS", "フィットネス"): {"スタイリスト": "インストラクター",
                             "アシスタント": "トレーニング補助", "受付": "フロント係"},
    # ---- 医療・福祉(MW)。国家資格名は業態で決まる ----
    ("MW", "クリニック"): {"医療スタッフ": "看護師", "介護スタッフ": "看護助手"},
    ("MW", "歯科"): {"医療スタッフ": "歯科衛生士", "介護スタッフ": "歯科助手"},
    ("MW", "調剤薬局"): {"医療スタッフ": "薬剤師", "介護スタッフ": "登録販売者",
                         "受付": "調剤事務", "事務": "調剤事務"},
    ("MW", "介護/福祉"): {"医療スタッフ": "看護師", "介護スタッフ": "介護士",
                          "受付": "生活相談員", "事務": "事務員"},
    ("MW", "治療院"): {"医療スタッフ": "柔道整復師", "介護スタッフ": "鍼灸師",
                       "受付": "受付", "事務": "事務員"},
    # ---- 教育(ED)----
    ("ED", "音楽教室"): {"講師": "音楽講師"},
    ("ED", "英会話"): {"講師": "語学講師"},
    ("ED", "学習塾"): {"講師": "塾講師"},
    ("ED", "予備校"): {"講師": "塾講師"},
    # ---- 娯楽(AM)----
    ("AM", "ゲームセンター"): {"フロア": "ゲームセンター店員"},
    ("AM", "カラオケ"): {"フロア": "カラオケ店員"},
    ("AM", "劇場・興行"): {"フロア": "劇場スタッフ", "受付": "劇場受付"},
    ("AM", "ライブハウス"): {"フロア": "ライブハウススタッフ"},
    ("AM", "スポーツ施設"): {"フロア": "スポーツ施設スタッフ", "受付": "フロント係"},
    # ---- 専門サービス(PS)。士業は業態で決まる ----
    ("PS", "税務・会計"): {"コンサルタント": "税理士", "ディレクター": "会計士",
                           "制作進行": "会計事務", "プランナー": "財務プランナー"},
    ("PS", "法務"): {"コンサルタント": "弁護士", "制作進行": "パラリーガル",
                     "ディレクター": "法務担当", "プランナー": "法務担当"},
    ("PS", "人材紹介"): {"コンサルタント": "キャリアアドバイザー", "営業": "人材営業",
                         "制作進行": "採用事務"},
    ("PS", "デザイン/ブランディング"): {"デザイナー": "グラフィックデザイナー",
                                        "ディレクター": "アートディレクター"},
    ("PS", "広告運用/制作"): {"プランナー": "広告プランナー",
                              "ディレクター": "広告ディレクター"},
    ("PS", "経営/ITコンサル"): {"コンサルタント": "経営コンサルタント"},
    # ---- 不動産(RE)----
    ("RE", "物品賃貸"): {"営業": "リース営業", "物件管理": "レンタル管理担当",
                         "事務": "事務員"},
    ("RE", "開発/PM"): {"物件管理": "開発プロジェクト担当"},
    # ---- 建設(CN)----
    ("CN", "土木"): {"設計": "土木設計士", "施工管理": "土木施工管理技士"},
    ("CN", "設備工事"): {"設計": "設備設計士", "施工管理": "設備施工管理技士"},
    ("CN", "内装/リフォーム"): {"設計": "内装設計士", "施工管理": "内装施工管理者"},
    # ---- 金融(FI)----
    ("FI", "証券"): {"アナリスト": "証券アナリスト", "営業": "証券営業"},
    ("FI", "投資/ファンド"): {"アナリスト": "運用アナリスト"},
    ("FI", "保険代理"): {"コンサルタント": "保険外交員", "営業": "保険外交員"},
    ("FI", "決済/フィンテック"): {"アナリスト": "データアナリスト"},
    # ---- 製造(MF)----
    ("MF", "印刷"): {"生産管理": "印刷オペレーター", "企画": "制作企画担当"},
    ("MF", "食品加工"): {"生産管理": "食品製造工"},
    ("MF", "アパレル生産管理"): {"生産管理": "アパレル生産管理担当"},
    # ---- サービス(SV)。★conf が名指ししていたのに名簿に 0 人だった語がここで生える ----
    ("SV", "警備"): {"オペレーション": "警備司令", "スタッフ": "警備員"},
    ("SV", "施設管理/メンテ"): {"オペレーション": "設備管理員", "スタッフ": "設備保守員"},
    ("SV", "BPO/事務代行"): {"オペレーション": "事務オペレーター", "スタッフ": "事務スタッフ"},
    ("SV", "人材派遣"): {"オペレーション": "派遣コーディネーター", "スタッフ": "派遣スタッフ"},
    ("SV", "各種代行"): {"オペレーション": "受託オペレーター", "スタッフ": "代行スタッフ"},
    # ---- 運輸(TR)----
    ("TR", "宅配拠点"): {"ドライバー": "宅配ドライバー", "倉庫管理": "仕分け作業員"},
    ("TR", "倉庫"): {"オペレーション": "荷役オペレーター"},
    # ---- 複合サービス(CS)----
    ("CS", "郵便局窓口"): {"窓口": "郵便窓口係"},
}

#: 学校の職員ロール → 職業名(教員はそのまま = 実在の職業名)。
_OCC_SCHOOL: dict[str, str] = {"職員": "学校事務職員"}


def occupation_for(industry_key: str, sector_detail: str, role: str) -> str:
    """台帳の (業種, 細分類, ロール) → 職業名。写像に無ければ**ロール名のまま**。"""
    role = str(role or "")
    sector = _OCC_BY_SECTOR.get((str(industry_key or ""), str(sector_detail or "")))
    if sector and role in sector:
        return sector[role]
    industry = _OCC_BY_INDUSTRY.get(str(industry_key or ""))
    if industry and role in industry:
        return industry[role]
    return role


# ---- L4 来訪セグメント(匿名合成: 目的×属性。実個人は入れない)----
_VISIT_PURPOSES = [
    ("観光・見物", 0.24), ("買い物", 0.26), ("飲食", 0.18), ("エンタメ・イベント", 0.12),
    ("ビジネス来訪", 0.10), ("友人と会う", 0.07), ("通院・用事", 0.03),
]
_L4_OCCS = [
    ("会社員", 0.34), ("学生", 0.16), ("販売・サービス", 0.12), ("主婦・主夫", 0.08),
    ("フリーランス", 0.07), ("経営者", 0.04), ("公務員", 0.03), ("無職・求職", 0.06),
    ("専門職", 0.10),
]

# ---- L5 役割ペルソナ(インフラ需要から人数を確定。docs/plans/persona-pool.md §4)----
# (role, occupation, フル人数, post 候補, duty_pattern, visitor)
_L5_ROLES = [
    ("駅員",        "駅員",        300, ["改札", "ホーム", "案内", "事務室"],
     {"days": "all", "rotates": True, "shift_hours": 8}, True),
    ("電車運転士",  "電車運転士",   60, ["山手線", "埼京線", "東横・副都心線", "田園都市・半蔵門線", "井の頭線"],
     {"days": "all", "rotates": True, "shift_hours": 8}, True),
    ("バス運転士",  "バス運転士",  120, ["東口ターミナル", "西口ターミナル", "南口", "マークシティ"],
     {"days": "all", "rotates": True, "shift_hours": 8}, True),
    ("タクシー運転手", "タクシー運転手", 350, ["道玄坂乗場", "宮益坂乗場", "駅前ロータリー", "流し(bbox内)"],
     {"days": "all", "rotates": True, "shift_hours": 10}, True),
    ("警察官",      "警察官",      140, ["渋谷警察署", "駅前交番", "宇田川交番", "神南交番", "巡回ユニット"],
     {"days": "all", "rotates": True, "shift_hours": 8}, True),
    ("配信者",      "配信者",       16, ["ハチ公前", "スクランブル交差点", "センター街", "宮下パーク"],
     {"days": "all", "rotates": False, "shift_hours": 6}, True),
    # ---- 街の顔 = 路上の生業(Wave 4 III-3。実装 src/society/street_life.py・既定 OFF)----
    # ★**必ず末尾に足す**: L5 の id は _L5_ROLES の並び順に振られる(idx_global)ので、
    #   途中に挿すと既存 L5 の id が全部ずれる。末尾追加なら既存 id と既存 rng 消費は不変。
    # ★人数の根拠(現実の実数 vs 名簿。過大に盛らない方針で下限側に寄せた):
    #   ティッシュ配り 30 … 公表統計は無い。配布は道路使用許可つきで駅出口・商店街入口の
    #     10 地点前後に集中し、目視で常時 10〜20 人規模。朝夕 2 交替で常時 15 人 = フル 30。
    #   ストリートミュージシャン 15 … 路上演奏の許可は事実上下りず黙認と退去の循環。
    #     夜間に 3〜4 地点 × 同時 2〜4 組 = 常時 8〜16 人。フル 15。
    #   キッチンカー営業者 6 … 許可された区画の出店枠は 1 日数台。フル 6(昼 1 帯のみ)。
    #   路上占い師 4 … 夜の裏通りに数人(定点観測の常連は片手で数えられる規模)。
    #   街頭演説者 4 … 政治的主体の本体は既存の議員 34 名。専業の街頭要員(政党スタッフ等)は
    #     少数なのでフル 4。選挙期の増員は本波では扱わない(module docstring の限界 5)。
    #   募金スタッフ 6 … 駅前の街頭募金は日により 1〜2 団体 × 3〜5 人。フル 6。
    #   路上生活者 80 … 区の昼間概数調査 74 人(23 区最多)、夜間推計 150〜200 人。
    #     ★**昼間実数側の 74 に寄せた 80** を採る(夜間推計は幅が大きく、盛ると
    #       「路上生活者だらけの渋谷」という現実に無い像を作るため)。
    #   路上支援員 4 … 区の巡回相談は少人数チーム。支援事業が世界に存在することを示す最小数。
    # ★visitor=False(この街で暮らしている)。路上生活者は commute も residence_line も持たない。
    ("ティッシュ配り", "ティッシュ配り", 30,
     ["駅ハチ公口", "駅東口", "駅西口", "商店街入口"],
     {"days": "weekday", "rotates": True, "shift_hours": 3}, False),
    ("ストリートミュージシャン", "ストリートミュージシャン", 15,
     ["ハチ公前", "センター街入口", "駅前広場", "公園通り"],
     {"days": "all", "rotates": False, "shift_hours": 5}, False),
    ("キッチンカー営業者", "キッチンカー営業者", 6,
     ["区役所前", "オフィス街の広場", "公園そばの空地"],
     {"days": "weekday", "rotates": False, "shift_hours": 3}, False),
    ("路上占い師", "路上占い師", 4,
     ["駅東口の路地", "道玄坂の裏通り", "宮益坂の路地"],
     {"days": "all", "rotates": False, "shift_hours": 4}, False),
    ("街頭演説者", "街頭演説者", 4,
     ["ハチ公前", "駅東口", "区役所前"],
     {"days": "all", "rotates": False, "shift_hours": 3}, False),
    ("募金スタッフ", "募金スタッフ", 6,
     ["駅前広場", "ハチ公前", "駅東口"],
     {"days": "all", "rotates": True, "shift_hours": 4}, False),
    ("路上生活者", "路上生活者", 80,
     ["公園のそば", "地下通路", "高架下", "駅の周辺"],
     {"days": "all", "rotates": False, "shift_hours": 0}, False),
    ("路上支援員", "路上支援員", 4,
     ["区の相談窓口", "巡回ルート"],
     {"days": "weekday", "rotates": False, "shift_hours": 2}, False),
    # ---- 車掌(Wave 4 II-1。★**死んでいた config の修復**)-------------------------- #
    # 監査事実: conf の transit_staff.bind.crew_occupations は既定で
    #   ["車掌", "電車運転士"] を指しているのに、**名簿に「車掌」が 1 人も居なかった**
    #   (_L5_ROLES に行が無い = 乗務員の半分が世界に存在しない設定キーだった)。
    #   その結果、ドア閉判断は電車運転士だけが担い、車内の巡回にいたっては担い手が
    #   原理的に居なかった。ここは新機能ではなく**設定と名簿の食い違いの修復**である。
    # ★**必ず末尾に足す**(上の注記と同じ理由): L5 の id は並び順に振られるので、
    #   途中に挿すと既存 L5 の id が全部ずれる。末尾追加なら既存 id と rng 消費は不変。
    # 人数の根拠(過大に盛らない方針で下限側):
    #   渋谷駅に乗り入れる 9 路線のうち、ワンマン運転でない = 車掌が乗務する系統は
    #   JR(山手・埼京)と東急・京王・メトロの一部。1 系統あたり当駅に関わる乗務員を
    #   常時 3〜5 人 × 2 交替と見て 40。既存の「電車運転士 60」より少なくしてあるのは
    #   ワンマン系統には運転士だけが乗るためで、**運転士 ≥ 車掌**という向きは現実と一致する。
    ("車掌", "車掌", 40,
     ["山手線", "埼京線", "東横・副都心線", "田園都市・半蔵門線", "井の頭線"],
     {"days": "all", "rotates": True, "shift_hours": 8}, True),
    # ---- 都市運営 = 見えない労働力(Wave 4 III-4。実装 src/society/city_ops.py・既定 OFF)----
    # ★**必ず末尾に足す**(上の 2 か所と同じ理由): L5 の id は _L5_ROLES の並び順に振られる
    #   (idx_global)ので、途中に挿すと既存 L5 の id が全部ずれる。末尾追加なら既存 id と
    #   既存 rng 消費は不変。
    # ★人数の根拠(現実の実数 vs 名簿。過大に盛らない方針で下限側に寄せた):
    #   清掃作業員 12 … 区の可燃ごみは 2 回/週の地区別収集。地図の範囲(駅中心の数百 m 四方)を
    #     格子 3×2 = 6 地区で回すとして、1 地区あたり 2 人(運転手 + 収集)= 12。
    #     ★区全体の清掃職員数ではなく、**この地図の範囲を回す班の人数**である。
    #   納品ドライバー 10 … 物流①(goods)の補充トリップは在庫が発注点を割った POI にだけ
    #     出る(実測で 1 日十数本の桁)。1 人が午前中に複数店を回る現実に合わせ、
    #     トリップ数より少ない 10 人。★車両の実体は作っていないので 1 人が同 step に
    #     複数のトリップを持ちうる(city_ops の docstring に明記した正直な近似)。
    #   夜間清掃員 10 … 地図の office 建物は v7 で 100 棟・旧地図で 33 棟。1 人 1 棟の
    #     担当(buildings_per_worker)で、夜勤 1 帯ぶんの最小構成として 10。
    #     ★台帳(build_orgs --night-shifts)の「夜間清掃」ロールが L2 に載っていればそちらが
    #       本筋で、この L5 役割は**台帳を再生成していないランでも機能が demo できる**ための
    #       フォールバックである(city_ops.night_cleaning.occupations が両方を引く)。
    #   救急隊員 6 … 区内の救急出動は実測 1 日 65 件だが、これは区全体(人口 23 万)の数字。
    #     本シムの規模では 1 隊(3 人)× 2 交替 = 6 が最小の常時稼働構成。
    #     ★既存名簿の「消防士」も救急の担い手候補に含める(crew_occupations)ので、
    #       公務員入り名簿のランではこの役割が 0 人でも連鎖は成立する。
    # ★visitor=False(この街で暮らしている)。
    ("清掃作業員", "清掃作業員", 12,
     ["北部の収集ルート", "中央の収集ルート", "南部の収集ルート", "深夜の事業系ルート"],
     {"days": "all", "rotates": True, "shift_hours": 6}, False),
    ("納品ドライバー", "納品ドライバー", 10,
     ["早朝の荷捌き場", "商店街の納品ルート", "飲食店の納品ルート"],
     {"days": "all", "rotates": True, "shift_hours": 7}, False),
    ("夜間清掃員", "夜間清掃員", 10,
     ["オフィスビル", "駅周辺のビル", "商業ビル"],
     {"days": "all", "rotates": False, "shift_hours": 7}, False),
    ("救急隊員", "救急隊員", 6,
     ["消防出張所", "救急待機所", "巡回ユニット"],
     {"days": "all", "rotates": True, "shift_hours": 8}, False),
    # ---- 消防(第109バッチ レーン FIX。★**名簿と conf の食い違いの修復**)------------ #
    # 監査事実(第108 縦煙の実測): ``incidents_env.fire.occupations`` の既定は「消防士」なのに、
    #   名簿に「消防士」が **1 人も居なかった**(_L5_ROLES に行が無い)。その結果、火災が起きても
    #   出場者が原理的に見つからず ``fire_dispatch`` は毎回 ``unstaffed=true`` になる。
    #   ``city_ops.ems.crew_occupations``(= 救急隊員 + 消防士)も片肺で回っていた
    #   (現実の東京消防庁では救急隊員も消防吏員 = 同じ署の人員である)。
    #   ここは新機能ではなく、車掌のときと同型の**設定と名簿の食い違いの修復**である。
    # ★**必ず末尾に足す**(上の 3 か所と同じ理由): L5 の id は _L5_ROLES の並び順に振られる
    #   ので、途中に挿すと既存 L5 の id が全部ずれる。末尾追加なら既存 id と rng 消費は不変。
    # ★人数の根拠(現実の実数 vs 名簿。過大に盛らない方針で**この地図の範囲**に絞った):
    #   消防士 45 …
    #     ① 渋谷消防署は本署 + 出張所 5(恵比寿・代々木・原宿・富ヶ谷・松濤)から成るが、
    #        **地図 v8 に載っている消防施設は「渋谷消防署」1 件だけ**(POI 実測)。出張所は
    #        bbox の外なので数えない(= 区全体の消防吏員数ではなく、この地図を守る本署の
    #        消火部隊だけを名簿に立てる。清掃作業員 12 と同じ「この範囲を回る班」の流儀)。
    #     ② 本署の消火部隊をポンプ隊 2 + はしご隊 1 = 3 隊、1 隊 5 人 = **15 人/当番**と置く。
    #     ③ 東京消防庁の消防署は**当番・非番・週休の 3 部制**なので 15 × 3 = 45。
    #     ★区全体の消防吏員は 200 人超の桁だが、それを名簿に載せると「地図の数百 m 四方に
    #       区全体の消防が常駐している街」という現実に無い像になるので採らない。
    #     ★救急隊員(上の 6 人)とは**別建て**にして二重計上しない。ただし
    #       ``city_ops.ems.crew_occupations`` は両方を引くので、救急の当直は実質両者が担う
    #       (現実の消防吏員の運用と一致する)。``incidents_env._fire_crew`` は救急出動中の
    #       個体を除くので、同じ人が同時に 2 つの現場へ行くことはない。
    # ★「警備員」は**足さない**(意図的な不作為): センサス較正台帳(build_orgs --census
    #   --night-shifts)が L2 に **常駐警備 3,875 人**を生やしていて、それがこの街で実際に
    #   働いている警備の実数である。L5 に「警備員」を別に立てると同じ職を二重に数えることに
    #   なる。conf 側は ``incidents_env.crowd.guard_occupations`` /
    #   ``world.facilities.responder_occupations`` に台帳ロール名(常駐警備・設備巡回)を
    #   並べて解決済みなので、名簿を増やす必要が無い。
    # ★visitor=False(この街で暮らしている)。
    ("消防士", "消防士", 45,
     ["渋谷消防署", "ポンプ隊", "はしご隊", "指揮隊"],
     {"days": "all", "rotates": True, "shift_hours": 8}, False),
]

# 役割ごとの persona 文の中核(既定は「{post}で{role}として働いている」)。
# ★路上生活者だけは「働いている」が事実に合わないので専用文にする。文面は**中立語のみ**
#   (蔑称・スティグマ語を 1 語も使わない = street_life.py の尊厳規約 1)。
_L5_ROLE_SENTENCE: dict[str, str] = {
    "路上生活者": "渋谷の{post}で寝起きし、路上で生活している。",
    "路上支援員": "渋谷区の路上生活者支援の仕事で、{post}を回って相談に応じている。",
}
# 議員(選挙なし・事前決定。occupation は tools.COUNCILOR_OCCS に合致)。
COUNCILOR_OCC = "議員"
N_COUNCILORS = 34            # 現実の渋谷区議会=34議席。config の assembly.size(=9)以上を満たす
_PARTIES = ["無所属", "区政クラブ", "みらい渋谷", "区民ネット", "生活者連合",
            "刷新の会", "共生フォーラム"]


def _weighted_choice_arr(rng: np.random.Generator, items, n: int):
    """(値, 重み) のリストから n 個を重み付き抽出(値配列で返す)。"""
    vals = [it[0] for it in items]
    w = np.array([it[1] for it in items], dtype=float)
    w /= w.sum()
    idx = rng.choice(len(vals), size=n, p=w)
    return np.array(vals, dtype=object)[idx]


def _seed_rng(master_seed: int, layer_code: int, part_idx: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        entropy=[int(master_seed), int(layer_code), int(part_idx)]))


# ------------------------------------------------------------------ 共通の個体差サンプラ(ベクトル化)
def _traits_block(rng: np.random.Generator, n: int):
    """traits(internal_locus/nfc/risk_tolerance)+ drive_threshold/fire_weight をベクトル化生成。

    値域は society.factors.registry と gen_personas.py を踏襲(N(0.5,0.18) clip[0,1]・裾確保)。"""
    locus = np.clip(rng.normal(0.5, 0.18, n), 0.0, 1.0)
    nfc = np.clip(rng.normal(0.5, 0.18, n), 0.0, 1.0)
    risk = np.clip(rng.normal(0.5, 0.18, n), 0.0, 1.0)
    # tail_frac=0.1 で 1 因子を上位裾[0.9,1.0]へ
    tail_hit = rng.random(n) < 0.1
    tail_which = rng.integers(0, 3, n)
    tail_val = rng.uniform(0.9, 1.0, n)
    locus = np.where(tail_hit & (tail_which == 0), tail_val, locus)
    nfc = np.where(tail_hit & (tail_which == 1), tail_val, nfc)
    risk = np.where(tail_hit & (tail_which == 2), tail_val, risk)
    thr = np.clip(rng.normal(0.60 - 0.20 * (nfc - 0.5) * 2, 0.08), 0.30, 0.85)
    fw = np.clip(rng.normal(0.50 + 0.30 * (locus - 0.5) * 2, 0.12), 0.15, 0.90)
    return locus, nfc, risk, thr, fw


def _names_block(rng: np.random.Generator, genders):
    """性別配列に応じた姓名(手続き生成)。姓∈_FAMILY・名∈_GIVEN_*。"""
    n = len(genders)
    fam = np.array(_FAMILY, dtype=object)[rng.integers(0, len(_FAMILY), n)]
    gf = np.array(_GIVEN_F, dtype=object)[rng.integers(0, len(_GIVEN_F), n)]
    gm = np.array(_GIVEN_M, dtype=object)[rng.integers(0, len(_GIVEN_M), n)]
    out = []
    for i in range(n):
        giv = gf[i] if genders[i] == "女" else gm[i]
        out.append(str(fam[i]) + str(giv))
    return out


def _bedtime_nonwork(rng: np.random.Generator, n):
    return ((22 * 60) + rng.integers(0, 24, n) * 10) % 1440


def _sleep_steps(rng: np.random.Generator, n):
    return rng.integers(39, 49, n)


def _depart_block(rng: np.random.Generator, n):
    evening = rng.random(n) < 0.8
    v = np.where(evening, rng.normal(1110, 45, n), rng.uniform(1320, 1440, n))
    v = np.round(v / 10.0) * 10
    return np.clip(v, 16 * 60, 23 * 60 + 50).astype(int)


def _lead_block(rng: np.random.Generator, n, is_student):
    mean = np.where(is_student, 30.0, 40.0)
    sd = np.where(is_student, 15.0, 20.0)
    return np.clip(np.round(rng.normal(mean, sd)), 10, 120).astype(int)


# ------------------------------------------------------------------ シャード書き出し
class ShardWriter:
    """層ごとに part-XXXX.jsonl を PART_SIZE 刻みで書き出し、件数と先頭行ハッシュを記録する。"""

    def __init__(self, root: Path, layer: str):
        self.dir = root / layer
        self.dir.mkdir(parents=True, exist_ok=True)
        self.layer = layer
        self.part_idx = 0
        self.buf: list[str] = []
        self.count = 0
        self.shards: list[dict] = []
        # 職業分布(PRES-B の検収用。meta.json へ出すだけでファイル本体は 1 バイトも変わらない)。
        self.occ: dict[str, int] = {}

    def add(self, rec: dict) -> None:
        self.buf.append(json.dumps(rec, ensure_ascii=False))
        self.count += 1
        o = str(rec.get("occupation", ""))
        self.occ[o] = self.occ.get(o, 0) + 1
        if len(self.buf) >= PART_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        name = f"part-{self.part_idx:04d}.jsonl"
        text = "\n".join(self.buf) + "\n"
        (self.dir / name).write_text(text, encoding="utf-8")
        h = hashlib.blake2b(self.buf[0].encode("utf-8"), digest_size=8).hexdigest()
        self.shards.append({"file": f"{self.layer}/{name}", "records": len(self.buf),
                            "first_row_blake2b": h})
        self.part_idx += 1
        self.buf = []

    def close(self) -> None:
        self._flush()


# ------------------------------------------------------------------ L1 住民(IPF)
def _ipf_joint(pop: dict, iters: int = 60) -> np.ndarray:
    ages = pop["age_bands"]; occs = pop["occupations"]
    genders = list(pop["gender"].keys())
    m_age = np.array([a["share"] for a in ages], float)
    m_gen = np.array([pop["gender"][g] for g in genders], float)
    m_occ = np.array([o["share"] for o in occs], float)
    m_age /= m_age.sum(); m_gen /= m_gen.sum(); m_occ /= m_occ.sum()
    joint = np.ones((len(ages), len(genders), len(occs)))
    hints = pop.get("age_x_occupation_hints", {})
    band_keys = [f"{a['band'][0]}-{a['band'][1]}" for a in ages]
    for oi, occ in enumerate(occs):
        hint = hints.get(occ["name"])
        if isinstance(hint, dict):
            for ai, key in enumerate(band_keys):
                joint[ai, :, oi] *= float(hint.get(key, 1.0))
    for _ in range(iters):
        joint *= (m_age / joint.sum(axis=(1, 2)))[:, None, None]
        joint *= (m_gen / joint.sum(axis=(0, 2)))[None, :, None]
        joint *= (m_occ / joint.sum(axis=(0, 1)))[None, None, :]
    return joint / joint.sum()


def gen_L1(writer: ShardWriter, n: int, seed: int, pop: dict) -> None:
    ages = pop["age_bands"]; occs = pop["occupations"]
    genders = list(pop["gender"].keys())
    joint = _ipf_joint(pop)
    flat = (joint.ravel() / joint.sum())
    idx_global = 0
    for part_idx in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        m = min(PART_SIZE, n - part_idx * PART_SIZE)
        rng = _seed_rng(seed, LC["L1"], part_idx)
        k = rng.choice(len(flat), size=m, p=flat)
        ai, gi, oi = np.unravel_index(k, joint.shape)
        los = np.array([ages[a]["band"][0] for a in ai])
        his = np.array([ages[a]["band"][1] for a in ai])
        age = (los + (rng.random(m) * (his - los + 1)).astype(int))
        gender = [genders[g] for g in gi]
        occ = [occs[o]["name"] for o in oi]
        names = _names_block(rng, gender)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        bedtime = _bedtime_nonwork(rng, m)
        slp = _sleep_steps(rng, m)
        bike = rng.random(m) < 0.15; car = rng.random(m) < 0.08
        hob = np.array(_HOBBIES, object)[rng.integers(0, len(_HOBBIES), m)]
        ton = np.array(_TONES, object)[rng.integers(0, len(_TONES), m)]
        for i in range(m):
            pid = f"L1_{idx_global:08d}"; idx_global += 1
            persona = (f"あなたは{names[i]}、{int(age[i])}歳の{occ[i]}({gender[i]}性)。"
                       f"渋谷の街に住んでいる。{hob[i]}が好きで、{ton[i]}。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L1", "presence": "resident",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": occ[i], "visitor": False, "commute": False,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedtime[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": bool(bike[i]), "has_car": bool(car[i]),
            })


# ------------------------------------------------------------------ 夜勤(Wave 4 III-1「夜間開放」)
# 台帳(scripts/build_orgs.py --night-shifts)が社に載せた ``night_shift`` を L2 の個体へ継承する。
# ★台帳に night_shift が 1 件も無ければ以下は全部 no-op = 従来のプールと 1 バイトも変わらない。
# ★決定論・**乱数を 1 draw も引かない**(枠の割当は末尾から数えるだけ・就寝時刻は退勤時刻の
#   純関数)。既存の rng 消費順を動かさないので、夜勤なし台帳での再生成は完全に同一出力。
def _hhmm_min(text, default: int = 0) -> int:
    """"HH:MM" → 分 of day(society/work._hhmm_to_min と同じ規約)。"""
    try:
        h, m = str(text).split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError, TypeError):
        return int(default)


def is_night_shift(sp: dict | None) -> bool:
    """その shift_pattern が日跨ぎ(夜勤)か = close < open。"""
    if not sp:
        return False
    return _hhmm_min(sp.get("close"), 0) < _hhmm_min(sp.get("open"), 0)


def night_slot_count(company: dict, k: int) -> int:
    """その社の従業者スロット k のうち夜勤に回す枠数(台帳 night_shift.share の丸め)。

    ★build_orgs.night_slot_count と同一の式。台帳生成器を import せずに済ませるため
      ここに写経してあり、tests/test_night_economy.py が両者の一致を機械で固定する。"""
    ns = company.get("night_shift")
    if not ns or k <= 0:
        return 0
    n = int(round(int(k) * float(ns.get("share", 0.0))))
    return max(0, min(int(k), n))


def night_bedtime_min(sp: dict, i: int) -> int:
    """夜勤者の就寝時刻 = **退勤(close)の 30 分後**から 10 分刻みで 1 時間ぶん散らす。

    L2 の bedtime_min は「帰宅トリガ」として使われる(persona.build_agent)ので、夜勤者は
    夕方ではなく**朝**に帰る。決定論(i の純関数)= rng を 1 draw も引かない。"""
    return (_hhmm_min(sp.get("close"), 6 * 60) + 30 + (int(i) % 6) * 10) % 1440


# ------------------------------------------------------------------ L2 域内従業者(需要駆動)
def _build_L2_slots(orgs: dict, fraction: float, occupations: bool = True):
    """組織台帳の employees / 教職員数から従業者スロット(org_id/role/occupation/shift/days)を展開。

    `occupations=True`(既定)で **職業名対応表**(PRES-B)を通し、台帳の
    (industry_key, sector_detail, role) から現実の職業名を付ける。False にすると
    従来どおり occupation = role(= 第108 までの名簿と 1 バイト一致)。
    ★夜勤スロットのロールは**写像を通さない**(conf が名指しで引く語なので。規律 ④)。
    """
    slots = []  # (org_id, role, occupation, shift_pattern_dict, days)
    for c in orgs["companies"]:
        emp = c["size"]["employees"]
        k = emp if fraction >= 1.0 else int(round(emp * fraction))
        if k <= 0:
            continue
        roles = c.get("roles") or ["スタッフ"]
        sp = c.get("shift_pattern", {})
        days = sp.get("days", "mon-fri")
        ikey, sector = c.get("industry_key", ""), c.get("sector_detail", "")
        # 夜勤枠は**末尾から**確保する(日勤側の role 巡回を 1 バイトも動かさないため)。
        # 台帳に night_shift が無ければ n_night=0 = 従来と完全に同一のスロット列。
        n_night = night_slot_count(c, k)
        for i in range(k):
            if n_night and i >= k - n_night:
                nsp = c["night_shift"]
                nroles = nsp.get("roles") or roles
                nrole = nroles[i % len(nroles)]
                slots.append((c["id"], nrole, nrole, nsp, nsp.get("days", days)))
                continue
            role = roles[i % len(roles)]
            occ = occupation_for(ikey, sector, role) if occupations else role
            slots.append((c["id"], role, occ, sp, days))
    # 学校の教職員(capacity から逆算: おおむね生徒12人に1人)
    for s in orgs["schools"]:
        staff = max(3, int(round(s["capacity"] / 12.0)))
        k = staff if fraction >= 1.0 else int(round(staff * fraction))
        if k <= 0:
            continue
        st_roles = [r for r in (s.get("roles") or []) if r != "学生"] or ["教員", "職員"]
        tt = s.get("timetable", {})
        sp = {"open": tt.get("start", "08:30"), "close": "17:00",
              "days": "mon-fri", "rotates": False}
        for i in range(k):
            role = st_roles[i % len(st_roles)]
            occ = (_OCC_SCHOOL.get(role, role) if occupations else role)
            slots.append((s["id"], role, occ, sp, "mon-fri"))
    return slots


def gen_L2(writer: ShardWriter, slots, seed: int) -> None:
    n = len(slots)
    idx_global = 0
    for part_idx in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        base = part_idx * PART_SIZE
        m = min(PART_SIZE, n - base)
        rng = _seed_rng(seed, LC["L2"], part_idx)
        female = rng.random(m) < 0.49
        gender = ["女" if f else "男" for f in female]
        # 就業層は 22-64 中心
        age = np.clip(rng.normal(38, 11, m), 18, 68).astype(int)
        names = _names_block(rng, gender)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        depart = _depart_block(rng, m)
        lead = _lead_block(rng, m, np.zeros(m, bool))
        gate = np.where(rng.random(m) < 0.87, "station", "edge")
        line = np.array(_RESIDENCE_LINES, object)[rng.integers(0, len(_RESIDENCE_LINES), m)]
        slp = _sleep_steps(rng, m)
        bike = rng.random(m) < 0.15; car = rng.random(m) < 0.08
        ton = np.array(_TONES, object)[rng.integers(0, len(_TONES), m)]
        for i in range(m):
            org_id, role, occ, sp, days = slots[base + i]
            pid = f"L2_{idx_global:08d}"; idx_global += 1
            # Wave 4 III-1: 夜勤スロットは「夜勤で通っている」+ 就寝時刻を退勤後へずらす。
            # 台帳に night_shift が無ければ is_night_shift は常に False = 従来と完全に同一。
            night = is_night_shift(sp)
            commuting = "夜勤で通勤している" if night else "通勤している"
            bedtime = night_bedtime_min(sp, i) if night else int(depart[i])
            persona = (f"あなたは{names[i]}、{int(age[i])}歳の{occ}({gender[i]}性)。"
                       f"{line[i]}に住んでいて、渋谷({role})に{commuting}。{ton[i]}。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L2", "presence": "workday_shift",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": occ, "visitor": True, "commute": True,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedtime), "sleep_steps": int(slp[i]),
                "has_bicycle": bool(bike[i]), "has_car": bool(car[i]),
                "arrival_lead_min": int(lead[i]), "commute_gateway": str(gate[i]),
                "residence_line": str(line[i]),
                "org_id": org_id, "role": role,
                "shift_pattern": {"open": sp.get("open", "09:00"),
                                  "close": sp.get("close", "18:00"),
                                  "days": days, "rotates": bool(sp.get("rotates", False))},
                "work_days": days,
            })


# ------------------------------------------------------------------ L3 定期来街(学生+常連)
_SCHOOL_OCC = {"区立小学校": ("小学生", (6, 12)), "区立中学校": ("中学生", (12, 15)),
               "小中一貫校": ("小中学生", (6, 15)), "高校": ("高校生", (15, 18)),
               "大学": ("大学生", (18, 23)), "専門学校": ("専門学生", (18, 24))}


def _build_L3_student_slots(orgs: dict, fraction: float):
    slots = []  # (school_id, occ, age_lo, age_hi, start)
    for s in orgs["schools"]:
        cap = s["capacity"]
        k = cap if fraction >= 1.0 else int(round(cap * fraction))
        if k <= 0:
            continue
        occ, (lo, hi) = _SCHOOL_OCC.get(s["school_type"], ("学生", (7, 18)))
        start = s.get("timetable", {}).get("start", "08:30")
        for _ in range(k):
            slots.append((s["id"], occ, lo, hi, start))
    return slots


def gen_L3_students(writer: ShardWriter, slots, seed: int) -> None:
    n = len(slots)
    idx_global = 0
    for part_idx in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        base = part_idx * PART_SIZE
        m = min(PART_SIZE, n - base)
        rng = _seed_rng(seed, LC["L3"], part_idx)
        female = rng.random(m) < 0.49
        gender = ["女" if f else "男" for f in female]
        names = _names_block(rng, gender)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        lead = _lead_block(rng, m, np.ones(m, bool))
        gate = np.where(rng.random(m) < 0.87, "station", "edge")
        line = np.array(_RESIDENCE_LINES, object)[rng.integers(0, len(_RESIDENCE_LINES), m)]
        slp = _sleep_steps(rng, m)
        bedt = _bedtime_nonwork(rng, m)
        for i in range(m):
            sid, occ, lo, hi, start = slots[base + i]
            age = int(lo + rng.integers(0, hi - lo + 1))
            pid = f"L3_{idx_global:08d}"; idx_global += 1
            persona = (f"あなたは{names[i]}、{age}歳の{occ}({gender[i]}性)。"
                       f"{line[i]}から渋谷の学校に通っている。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L3", "presence": "cadence",
                "name": names[i], "age": age, "gender": gender[i],
                "occupation": occ, "visitor": True, "commute": True,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": bool(rng.random() < 0.2), "has_car": False,
                "arrival_lead_min": int(lead[i]), "commute_gateway": str(gate[i]),
                "residence_line": str(line[i]),
                "org_id": sid, "role": "学生",
                "visit_cadence": "school_day", "subtype": "student",
            })


def gen_L3_regulars(writer: ShardWriter, n: int, seed: int, part_offset: int) -> None:
    idx_global = 0
    for p in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        m = min(PART_SIZE, n - p * PART_SIZE)
        rng = _seed_rng(seed, LC["L3"], 1000 + p)  # 学生パートと名前空間を分離
        female = rng.random(m) < 0.52
        gender = ["女" if f else "男" for f in female]
        age = np.clip(rng.normal(34, 13, m), 16, 78).astype(int)
        names = _names_block(rng, gender)
        occ = _weighted_choice_arr(rng, _L4_OCCS, m)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        line = np.array(_RESIDENCE_LINES, object)[rng.integers(0, len(_RESIDENCE_LINES), m)]
        cad = rng.integers(2, 5, m)   # 週2-4回
        slp = _sleep_steps(rng, m)
        bedt = _bedtime_nonwork(rng, m)
        hob = np.array(_HOBBIES, object)[rng.integers(0, len(_HOBBIES), m)]
        for i in range(m):
            pid = f"L3reg_{idx_global:08d}"; idx_global += 1
            persona = (f"あなたは{names[i]}、{int(age[i])}歳の{occ[i]}({gender[i]}性)。"
                       f"{line[i]}に住み、週{int(cad[i])}回ほど渋谷に通う常連。{hob[i]}が好き。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L3", "presence": "cadence",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": str(occ[i]), "visitor": True, "commute": False,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": bool(rng.random() < 0.15), "has_car": bool(rng.random() < 0.08),
                "residence_line": str(line[i]),
                "visit_cadence": f"weekly_{int(cad[i])}", "subtype": "regular",
            })


# ------------------------------------------------------------------ L4 非定期来街(回転主層)
def gen_L4(writer: ShardWriter, n: int, seed: int) -> None:
    idx_global = 0
    for part_idx in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        m = min(PART_SIZE, n - part_idx * PART_SIZE)
        rng = _seed_rng(seed, LC["L4"], part_idx)
        female = rng.random(m) < 0.52
        gender = ["女" if f else "男" for f in female]
        age = np.clip(rng.normal(36, 14, m), 15, 82).astype(int)
        names = _names_block(rng, gender)
        occ = _weighted_choice_arr(rng, _L4_OCCS, m)
        purpose = _weighted_choice_arr(rng, _VISIT_PURPOSES, m)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        is_foreign = rng.random(m) < 0.15
        party = 1 + rng.integers(0, 5, m)  # 同行人数 1-5
        # 来訪確率/日(low・log-uniform): 大多数は稀。~[0.003, 0.06]
        visit_rate = np.exp(rng.uniform(np.log(0.003), np.log(0.06), m))
        revisit = rng.random(m) < 0.10
        slp = _sleep_steps(rng, m)
        bedt = _bedtime_nonwork(rng, m)
        for i in range(m):
            pid = f"L4_{idx_global:08d}"; idx_global += 1
            tag = "訪日外国人" if is_foreign[i] else str(occ[i])
            persona = (f"あなたは{names[i]}、{int(age[i])}歳({gender[i]}性)。"
                       f"{purpose[i]}のため渋谷を訪れる{tag}。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L4", "presence": "stochastic",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": str(occ[i]), "visitor": True, "commute": False,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": False, "has_car": bool(rng.random() < 0.05),
                "visit_purpose": str(purpose[i]),
                "visit_rate": round(float(visit_rate[i]), 4),
                "is_foreign": bool(is_foreign[i]), "party_size": int(party[i]),
                "revisit": bool(revisit[i]),
            })


# ------------------------------------------------------------------ L5 役割ペルソナ + 議員
def gen_L5(writer: ShardWriter, seed: int, fraction: float):
    """役割ペルソナ(駅員/運転士/タクシー/警察/配信者)+ 議員。councilors はフル 34 で固定。"""
    rng = _seed_rng(seed, LC["L5"], 0)
    councilors = []          # 標準スキーマの議員(personas_councilors.json 用に返す)
    idx_global = 0

    # --- 役割(インフラ需要)---
    for role, occ, full_n, posts, duty, is_visitor in _L5_ROLES:
        k = full_n if fraction >= 1.0 else max(1, int(round(full_n * fraction)))
        female = rng.random(k) < 0.35
        gender = ["女" if f else "男" for f in female]
        age = np.clip(rng.normal(40, 11, k), 20, 66).astype(int)
        names = _names_block(rng, gender)
        locus, nfc, risk, thr, fw = _traits_block(rng, k)
        line = np.array(_RESIDENCE_LINES, object)[rng.integers(0, len(_RESIDENCE_LINES), k)]
        slp = _sleep_steps(rng, k)
        bedt = _bedtime_nonwork(rng, k)
        for i in range(k):
            post = posts[i % len(posts)]
            pid = f"L5_{idx_global:08d}"; idx_global += 1
            # 役割別の中核文(既定 = 従来の「{post}で{role}として働いている」= バイト一致)。
            core = _L5_ROLE_SENTENCE.get(role, "渋谷の{post}で{role}として働いている。")
            persona = (f"あなたは{names[i]}、{int(age[i])}歳の{occ}({gender[i]}性)。"
                       + core.format(post=post, role=role)
                       + "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L5", "presence": "duty",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": occ, "visitor": bool(is_visitor),
                "commute": bool(is_visitor),
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": False, "has_car": bool(rng.random() < 0.1),
                "residence_line": str(line[i]) if is_visitor else "",
                "role": role, "post": post, "duty_pattern": dict(duty),
            })

    # --- 議員(選挙なし・事前決定。フル 34 固定)---
    # ★専用の乱数ストリーム(part_index=1)+ 固定 id で fraction 非依存にする
    #   = 議員名簿(コミット対象)が --fraction に関わらず同一(検収の安定性)。
    crng = _seed_rng(seed, LC["L5"], 1)
    k = N_COUNCILORS
    female = crng.random(k) < 0.35
    gender = ["女" if f else "男" for f in female]
    age = np.clip(crng.normal(52, 9, k), 30, 74).astype(int)
    names = _names_block(crng, gender)
    locus, nfc, risk, thr, fw = _traits_block(crng, k)
    slp = _sleep_steps(crng, k)
    bedt = _bedtime_nonwork(crng, k)
    car = crng.random(k) < 0.3
    for i in range(k):
        party = _PARTIES[i % len(_PARTIES)]
        pid = f"L5c_{i + 1:03d}"       # 議員は固定 id(fraction 非依存)
        seat_id = f"seat_{i + 1:02d}"
        persona = (f"あなたは{names[i]}、{int(age[i])}歳の渋谷区議会議員({gender[i]}性)。"
                   f"会派は{party}。渋谷に住み、区政に取り組んでいる。"
                   "自分の言葉で自然に、短く話す。")
        rec = {
            "id": pid, "layer": "L5", "presence": "resident",
            "name": names[i], "age": int(age[i]), "gender": gender[i],
            "occupation": COUNCILOR_OCC, "visitor": False, "commute": False,
            "persona": persona,
            "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                       "risk_tolerance": float(risk[i])},
            "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
            "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
            "has_bicycle": False, "has_car": bool(car[i]),
            "role": "議員", "seat_id": seat_id, "party": party,
        }
        writer.add(rec)
        # 議員名簿は標準スキーマ(拡張キーは残しても build_agent は無視する)
        councilors.append(rec)
    return councilors


# ------------------------------------------------------------------ メイン
def scan_existing_parts(out_dir: Path) -> set:
    """出力先に**いま在る** part-*.jsonl の相対名(``<layer>/part-XXXX.jsonl``)を集める。

    第109 レーン D1 の発見: ビルダは part を**上書きするだけ**で古い part を消さない。
    層が縮む再生成(例: L2 が 224,240 → 180,000 に減る)を掛けると、前回の末尾 part が
    ディレクトリに残る。影響の切り分け:

      - シム本体(``world/pool.py`` の ``PoolStore``)は ``meta.json`` の ``shards`` しか
        読まない = **名簿は汚れない**(幽霊 id は present 判定にも id_of にも入らない)。
      - ところが同じビルダの ``_collect_llm_targets`` は層ディレクトリを **glob する**ので、
        ``llm_targets.json`` には幽霊 id が**入り込む**(深いペルソナ化の対象名簿が汚れる)。
      - ディレクトリを直に舐める外部ツール(集計スクリプト・人間の ``wc -l``)も同じ。

    生成の**前**に採るのが要点で、そうすれば「今回書いたもの」との差が管理外 part そのもの
    になる(今回書いた part を巻き添えで消す事故が原理的に起きない)。
    """
    out: set = set()
    if not out_dir.is_dir():
        return out
    for layer_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        for f in sorted(layer_dir.glob("part-*.jsonl")):
            out.add(f"{layer_dir.name}/{f.name}")
    return out


def build_pool(out_dir: Path, seed: int, fraction: float,
               orgs: dict, pop: dict, total_target: int,
               orgs_file: str = DEFAULT_ORGS_FILE, clean: bool = False,
               occupations: bool = True):
    """プールを生成する。``clean=True`` で**管理外の古い part を削除**する(既定は警告のみ)。

    ``occupations=False`` で職業名対応表(PRES-B)を通さない = 第108 までの名簿と
    1 バイト一致(occupation = 台帳ロール名)。回帰比較用の口。
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_parts = scan_existing_parts(out_dir)      # ★書き込み前に採る(D1)

    # 層別目標
    n_L1 = int(round(30_000 * fraction))
    L2_slots = _build_L2_slots(orgs, fraction, occupations=occupations)
    n_L2 = len(L2_slots)
    L3_student_slots = _build_L3_student_slots(orgs, fraction)
    n_L3s = len(L3_student_slots)
    n_L3r = int(round(20_000 * fraction))
    # L5 件数を先に確定(L4=残り の計算に必要)
    n_L5 = N_COUNCILORS + sum(
        (full if fraction >= 1.0 else max(1, int(round(full * fraction))))
        for _r, _o, full, _p, _d, _v in _L5_ROLES)
    n_fixed = n_L1 + n_L2 + n_L3s + n_L3r + n_L5
    n_L4 = max(0, int(round(total_target * fraction)) - n_fixed)

    writers: dict[str, ShardWriter] = {}
    timings: dict[str, float] = {}

    def _run(layer, fn):
        w = writers.setdefault(layer, ShardWriter(out_dir, layer))
        s = time.time()
        fn(w)
        timings[layer] = timings.get(layer, 0.0) + (time.time() - s)

    _run("L1", lambda w: gen_L1(w, n_L1, seed, pop))
    _run("L2", lambda w: gen_L2(w, L2_slots, seed))
    _run("L3", lambda w: gen_L3_students(w, L3_student_slots, seed))
    _run("L3", lambda w: gen_L3_regulars(w, n_L3r, seed, part_offset=0))
    _run("L4", lambda w: gen_L4(w, n_L4, seed))
    councilors = []

    def _run_L5(w):
        nonlocal councilors
        councilors = gen_L5(w, seed, fraction)
    _run("L5", _run_L5)

    for w in writers.values():
        w.close()

    layer_counts = {ly: w.count for ly, w in writers.items()}
    total = sum(layer_counts.values())
    shards = [s for w in writers.values() for s in w.shards]
    # 職業分布(PRES-B の検収用。多い順・全数)。名簿本体は 1 バイトも変わらない。
    occ_all: dict[str, int] = {}
    for w in writers.values():
        for k, v in w.occ.items():
            occ_all[k] = occ_all.get(k, 0) + v
    occ_sorted = dict(sorted(occ_all.items(), key=lambda kv: (-kv[1], kv[0])))

    # ---- 管理外(stale)part の始末(第109 レーン D1)------------------------------
    # 既定は**破壊的でない**: 消さずに警告し、meta へ「meta.shards から参照していない」
    # 事実を明示して残す(再生成のたびに黙って消すと、取り違えた --out を指定した事故が
    # 復旧不能になる)。--clean を明示したときだけ削除する。
    # ★ここに置くのは `_collect_llm_targets`(層ディレクトリを glob する)の**前**でなければ
    #   ならない。後ろに置くと --clean を付けても llm_targets.json に幽霊 id が残る。
    produced = {s["file"] for s in shards}
    stale = sorted(pre_parts - produced)
    removed: list = []
    if stale and clean:
        for rel in stale:
            (out_dir / rel).unlink()
        removed, stale = stale, []
    elif stale:
        print(f"[warn] 管理外の古い part が {len(stale)} 個残っている。"
              f" 名簿(meta.shards → PoolStore)は汚れないが、"
              f" llm_targets.json は層ディレクトリを glob するので幽霊 id が混ざる。"
              f" 消すには --clean を付けて再実行する:", file=sys.stderr)
        for rel in stale:
            print(f"[warn]   {rel}", file=sys.stderr)

    # ---- llm_targets(深いペルソナ化=LLM上塗り対象: L5 全員 + L1/L3常連の一部)----
    llm_targets = _collect_llm_targets(out_dir, seed)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/build_persona_pool.py",
        "seed": seed, "fraction": fraction,
        "total_target": total_target, "total_generated": total,
        "layer_counts": layer_counts,
        "layer_targets": {"L1": n_L1, "L2": n_L2, "L3": n_L3s + n_L3r,
                          "L4": n_L4, "L5": n_L5},
        "councilors": len(councilors), "councilor_occupation": COUNCILOR_OCC,
        "presence_keys": ["resident", "workday_shift", "cadence", "stochastic", "duty"],
        "base_schema_ref": "data/personas_300_civic.json",
        "schema_extensions": [
            "id", "layer", "presence", "org_id", "role", "shift_pattern", "work_days",
            "visit_cadence", "subtype", "visit_purpose", "visit_rate", "is_foreign",
            "party_size", "revisit", "post", "duty_pattern", "seat_id", "party"],
        "sources": {
            "organizations": orgs_file,
            "population_marginals": "data/shibuya_population.json"},
        "organizations_meta": {k: orgs.get("meta", {}).get(k)
                               for k in ("map", "mode", "night_shifts", "seed")},
        "employees_ledger_total": sum(c["size"]["employees"] for c in orgs["companies"]),
        # ---- 職業多様性(PRES-B。第109バッチ レーン PRES-Pool)----------------------
        # occupation_map=false は「台帳ロール名をそのまま職業名にする」第108 までの形。
        "occupation_map": bool(occupations),
        "occupations_distinct": len(occ_sorted),
        "occupations": occ_sorted,
        "llm_targets_count": len(llm_targets),
        "elapsed_sec": round(time.time() - t0, 2),
        "layer_elapsed_sec": {k: round(v, 2) for k, v in timings.items()},
        "shards": shards,
        # 第109 レーン D1: 前回の生成が残した管理外 part(**shards からは参照しない**)と、
        # --clean で実際に消したもの。どちらも空リストが既定 = 通常の再生成では出ない。
        "stale_shards": stale,
        "stale_shards_removed": removed,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta, councilors


def _collect_llm_targets(out_dir: Path, seed: int) -> list[str]:
    """LLM 上塗り対象 id を層別ファイルから収集(L5 全員 + L1 10% + L3常連 全員)。~2-3%。"""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 999]))
    targets: list[str] = []
    # L5 全員
    for f in sorted((out_dir / "L5").glob("part-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                targets.append(json.loads(line)["id"])
    # L3 常連(subtype=regular)全員
    for f in sorted((out_dir / "L3").glob("part-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            if r.get("subtype") == "regular":
                targets.append(r["id"])
    # L1 の 10%
    l1_ids = []
    for f in sorted((out_dir / "L1").glob("part-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                l1_ids.append(json.loads(line)["id"])
    if l1_ids:
        k = max(1, int(round(len(l1_ids) * 0.10)))
        sel = rng.choice(len(l1_ids), size=k, replace=False)
        targets.extend(sorted(l1_ids[i] for i in sel))
    (out_dir / "llm_targets.json").write_text(
        json.dumps({"note": "深いペルソナ化(LLM上塗り)対象の id。生成は後日(本選前)。"
                            "L5全員 + L3常連 + L1の10%。",
                    "seed": seed, "count": len(targets), "ids": targets},
                   ensure_ascii=False), encoding="utf-8")
    return targets


def _write_councilors_json(councilors: list[dict], seed: int,
                           path: Path | None = None) -> Path:
    """議員名簿(コミット対象・小)。標準スキーマに整形(from_roster で着席可能)。"""
    path = path or (REPO_ROOT / "data" / "personas_councilors.json")
    meta = {"seed": seed, "generator": "scripts/build_persona_pool.py",
            "note": "名簿制議会(選挙なし・事前決定)。occupation=議員 は society.tools.COUNCILOR_OCCS "
                    "に合致。conf の institution_routes.assembly.from_roster=true で着席する。",
            "count": len(councilors), "occupation": COUNCILOR_OCC}
    path.write_text(json.dumps({"meta": meta, "personas": councilors},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="ペルソナ100万プールの決定論生成(W2 P5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="層別目標に掛ける倍率(1.0=フル~100万。テストは小さく)")
    ap.add_argument("--total", type=int, default=1_000_000,
                    help="fraction=1.0 での目標総数(既定 100万)")
    ap.add_argument("--out", default="data/persona_pool",
                    help="出力ディレクトリ(既定 data/persona_pool・.gitignore 済み)")
    ap.add_argument("--no-councilors-json", action="store_true",
                    help="data/personas_councilors.json を書かない")
    ap.add_argument("--orgs", default=DEFAULT_ORGS_FILE,
                    help="L2/L3学生の需要源になる組織台帳 JSON(既定=正準 11k 台帳)。"
                         "センサス較正台帳を使うなら "
                         "--orgs data/organizations_shibuya_census.json")
    ap.add_argument("--clean", action="store_true",
                    help="出力先に残った**管理外の古い part**(前回の生成物で今回は"
                         "書き直されなかったもの)を削除する。既定は削除せず警告のみ"
                         "(meta.shards から参照されないので中身は汚れない)。")
    ap.add_argument("--no-occupation-map", action="store_true",
                    help="職業名対応表(PRES-B)を通さず occupation = 台帳ロール名にする"
                         "(第108 までの名簿と 1 バイト一致。回帰比較用)。")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    orgs_path = Path(args.orgs)
    if not orgs_path.is_absolute():
        orgs_path = REPO_ROOT / orgs_path
    orgs = json.loads(orgs_path.read_text(encoding="utf-8"))
    pop = json.loads((REPO_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))

    meta, councilors = build_pool(out_dir, args.seed, args.fraction, orgs, pop, args.total,
                                  orgs_file=str(args.orgs).replace("\\", "/"),
                                  clean=args.clean,
                                  occupations=not args.no_occupation_map)

    if not args.no_councilors_json:
        cpath = _write_councilors_json(councilors, args.seed)
        print(f"written councilors: {cpath} ({len(councilors)})")

    print(f"pool written: {out_dir}")
    print(f"  total={meta['total_generated']:,}  seed={args.seed}  "
          f"fraction={args.fraction}  elapsed={meta['elapsed_sec']}s")
    for ly in ("L1", "L2", "L3", "L4", "L5"):
        c = meta["layer_counts"].get(ly, 0)
        print(f"  {ly}: {c:,}  ({meta['layer_elapsed_sec'].get(ly, 0)}s)")
    print(f"  councilors={meta['councilors']}  llm_targets={meta['llm_targets_count']:,}")
    print(f"  occupations={meta['occupations_distinct']} 種"
          f"  (occupation_map={meta['occupation_map']})")
    for occ, n in list(meta["occupations"].items())[:10]:
        print(f"    {occ}: {n:,}")
    if meta["stale_shards_removed"]:
        print(f"  stale parts removed (--clean): {len(meta['stale_shards_removed'])}")
    elif meta["stale_shards"]:
        print(f"  stale parts left in place: {len(meta['stale_shards'])}"
              f"  (meta.shards 非参照。--clean で削除)")


if __name__ == "__main__":
    main()
