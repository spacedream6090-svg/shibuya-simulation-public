"""組織台帳(架空)の生成 + エージェント→組織の配属(バッチE)。

エンジン(src/)は一切編集しない。本スクリプトは data/ 配下の JSON を生成するだけ:
  - data/organizations_shibuya.json     架空の組織台帳(会社 約40社 + 学校 約10校)
  - data/org_assignments_80.json        personas_80.json の配属
  - data/org_assignments_100_inflow.json personas_100_inflow.json の配属

設計根拠 = docs/research/shibuya-organizations.md(産業構成・教育課程・時間構造の出典)。
統合仕様 = docs/design-candidates/org-integration-spec.md。

倫理制約 R17: 実在の企業名・学校名はデータに入れない。名称はすべて架空。POI への束縛は
地図ノード(node/座標)参照のみで行い、OSM に含まれる実在店名は台帳の識別子にしない。

決定論: 組織台帳は (map, seed) から、配属は (roster, agent_index, name, occupation) の
安定ハッシュ(sha256)から一意に決まる。LLM は一切呼ばない。

使い方:
    python scripts/build_orgs.py                       # 台帳 + 既知2名簿の配属を再生成
    python scripts/build_orgs.py --roster data/personas_80.json   # 1名簿だけ再生成
    python scripts/build_orgs.py --map data/shibuya_osm_wide_20250401.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
DEFAULT_MAP = DATA / "shibuya_osm.json"
DEFAULT_ROSTERS = [DATA / "personas_80.json", DATA / "personas_100_inflow.json"]


# ----------------------------------------------------------------------------
# 決定論ハッシュ(Python 組込み hash は起動毎に salt されるので使わない)
# ----------------------------------------------------------------------------
def _h(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)


def _pick(seq, *parts):
    """seq から安定ハッシュで1件選ぶ。"""
    return seq[_h(*parts) % len(seq)]


# ----------------------------------------------------------------------------
# 組織台帳の仕様(架空。構成比は経済センサス準拠 = docs/research/shibuya-organizations.md)
#   size band: 従業者規模。都市部は小規模事業所が多数 + 少数の中〜大規模(§1.4)。
#   wage_tier: economy.WAGE_CAT のキー(会社員/店員/自営)に対応。
#   poi_cat  : 束縛する地図 POI カテゴリ。None = 固定拠点なし(自営・ギグ)。
# ----------------------------------------------------------------------------
# 会社(架空名。実在地名は可)。roles の先頭 = specialist、generalist は 会社員 の就く職種。
COMPANIES: list[dict] = [
    # --- 情報通信/IT(ビットバレー。§1.2 で渋谷は都内最多の IT 集積) 8社 ---
    dict(id="co_it_01", name="(株)スクランブル計画", key="IT", detail="業務SaaS",
         products=["勤怠・労務管理SaaSの開発提供", "業務効率化ツールの受託開発"],
         emp=90, band="50-99", roles=["エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"],
         out=["software", "service"]),
    dict(id="co_it_02", name="(株)ハチ公ラボ", key="IT", detail="受託開発/DX",
         products=["企業のDX受託開発", "Web/業務システム構築"],
         emp=45, band="10-49", roles=["エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"],
         out=["software", "service"]),
    dict(id="co_it_03", name="センターガイ・ゲームス(株)", key="IT", detail="ゲーム開発",
         products=["スマホゲームの開発・運営", "ゲーム内課金サービス"],
         emp=160, band="100-299", roles=["エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"],
         out=["game", "service"]),
    dict(id="co_it_04", name="(株)道玄坂ソフトウェア", key="IT", detail="クラウド/API",
         products=["クラウドインフラ/API提供", "SREコンサルティング"],
         emp=70, band="50-99", roles=["エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"],
         out=["software", "service"]),
    dict(id="co_it_05", name="(株)宮益フィンテック", key="IT", detail="フィンテックSaaS",
         products=["決済・請求SaaSの開発提供", "会計連携API"],
         emp=55, band="50-99", roles=["エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"],
         out=["software", "service"]),
    dict(id="co_it_06", name="(株)神南データワークス", key="IT", detail="データ分析/AI",
         products=["データ分析・AIモデルの受託", "BIダッシュボード提供"],
         emp=35, band="10-49", roles=["エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"],
         out=["software", "service"]),
    dict(id="co_it_07", name="(株)公園通りメディア", key="IT", detail="Webメディア/CMS",
         products=["Webメディア運営", "CMSプラットフォーム提供"],
         emp=28, band="10-49", roles=["エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"],
         out=["content", "service"]),
    dict(id="co_it_08", name="(株)渋谷ヘルスケアIT", key="IT", detail="医療系SaaS",
         products=["医療・健康管理SaaSの開発提供", "オンライン診療支援"],
         emp=48, band="10-49", roles=["エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"],
         out=["software", "service"]),
    # --- 広告・コンテンツ制作 5社 ---
    dict(id="co_ad_01", name="(株)スクランブル・クリエイティブ", key="AD", detail="広告運用/クリエイティブ",
         products=["デジタル広告の運用", "クリエイティブ制作"],
         emp=60, band="50-99", roles=["デザイナー", "プランナー", "ディレクター", "営業", "制作進行"],
         out=["ad", "content"]),
    dict(id="co_ad_02", name="(株)ミヤシタ映像制作", key="AD", detail="映像/動画制作",
         products=["映像・動画コンテンツ制作", "配信番組の企画制作"],
         emp=30, band="10-49", roles=["デザイナー", "プランナー", "ディレクター", "営業", "制作進行"],
         out=["content", "ad"]),
    dict(id="co_ad_03", name="(株)道玄坂デザイン事務所", key="AD", detail="グラフィック/ブランディング",
         products=["ブランディング・グラフィックデザイン", "パッケージ/ロゴ制作"],
         emp=18, band="10-49", roles=["デザイナー", "プランナー", "ディレクター", "営業", "制作進行"],
         out=["design", "content"]),
    dict(id="co_ad_04", name="(株)センガイ・アドテク", key="AD", detail="広告配信/アドテク",
         products=["広告配信プラットフォーム運用", "アドテク支援"],
         emp=42, band="10-49", roles=["デザイナー", "プランナー", "ディレクター", "営業", "制作進行"],
         out=["ad", "service"]),
    dict(id="co_ad_05", name="(株)渋谷コンテンツスタジオ", key="AD", detail="SNS/インフルエンサー運用",
         products=["SNS運用代行", "インフルエンサーマーケティング"],
         emp=25, band="10-49", roles=["デザイナー", "プランナー", "ディレクター", "営業", "制作進行"],
         out=["content", "ad"]),
    # --- 卸売・小売/アパレル 7社(shop) ---
    dict(id="co_ap_01", name="(株)渋谷ファッションワークス", key="AP", detail="アパレル企画/EC/店舗",
         products=["アパレルの企画・EC・店舗販売"],
         emp=120, band="100-299", roles=["販売スタッフ", "店長", "バイヤー", "EC運営", "VMD"],
         out=["goods", "retail"]),
    dict(id="co_ap_02", name="(株)センター街アパレル", key="AP", detail="セレクトショップ",
         products=["セレクトショップ運営", "衣料・服飾雑貨販売"],
         emp=40, band="10-49", roles=["販売スタッフ", "店長", "バイヤー", "EC運営", "VMD"],
         out=["goods", "retail"]),
    dict(id="co_ap_03", name="(株)ヒカリエ・スタイル", key="AP", detail="婦人服企画・小売",
         products=["婦人服の企画・小売"],
         emp=65, band="50-99", roles=["販売スタッフ", "店長", "バイヤー", "EC運営", "VMD"],
         out=["goods", "retail"]),
    dict(id="co_ap_04", name="(株)公園通りセレクト", key="AP", detail="雑貨・ライフスタイル小売",
         products=["生活雑貨・ライフスタイル用品の小売"],
         emp=22, band="10-49", roles=["販売スタッフ", "店長", "バイヤー", "EC運営", "VMD"],
         out=["goods", "retail"]),
    dict(id="co_ap_05", name="(株)宮下ストア", key="AP", detail="スニーカー/ストリート小売",
         products=["スニーカー・ストリートウェア販売"],
         emp=18, band="10-49", roles=["販売スタッフ", "店長", "バイヤー", "EC運営", "VMD"],
         out=["goods", "retail"]),
    dict(id="co_ap_06", name="(株)渋谷古着コレクティブ", key="AP", detail="古着/リユース",
         products=["古着・リユース衣料の買取と販売"],
         emp=14, band="10-49", roles=["販売スタッフ", "店長", "バイヤー", "EC運営", "VMD"],
         out=["goods", "retail"]),
    dict(id="co_ap_07", name="(株)道玄坂コスメ&ビューティ小売", key="AP", detail="コスメ物販",
         products=["化粧品・美容雑貨の小売"],
         emp=20, band="10-49", roles=["販売スタッフ", "店長", "バイヤー", "EC運営", "VMD"],
         out=["goods", "retail"]),
    # --- 飲食・カフェ 6社(food) ---
    dict(id="co_ca_01", name="(株)スクランブル珈琲", key="CA", detail="スペシャルティコーヒー",
         products=["スペシャルティコーヒーの提供", "カフェ運営"],
         emp=35, band="10-49", roles=["バリスタ", "ホール", "キッチン", "店長"],
         out=["food", "service"]),
    dict(id="co_ca_02", name="(株)渋谷ベーカリーカフェ", key="CA", detail="ベーカリーカフェ",
         products=["ベーカリー・カフェの運営", "焼きたてパンの販売"],
         emp=28, band="10-49", roles=["バリスタ", "ホール", "キッチン", "店長"],
         out=["food", "service"]),
    dict(id="co_ca_03", name="(株)センガイ・ダイニング", key="CA", detail="カフェ&ビストロ",
         products=["カフェ&ビストロの運営", "ランチ・ディナー提供"],
         emp=45, band="10-49", roles=["バリスタ", "ホール", "キッチン", "店長"],
         out=["food", "service"]),
    dict(id="co_ca_04", name="(株)ハチ公ティースタンド", key="CA", detail="ティースタンド",
         products=["ティー・ドリンクのテイクアウト販売"],
         emp=16, band="10-49", roles=["バリスタ", "ホール", "キッチン", "店長"],
         out=["food", "service"]),
    dict(id="co_ca_05", name="(株)道玄坂フードサービス", key="CA", detail="定食/ランチ",
         products=["定食・ランチ食堂の運営"],
         emp=22, band="10-49", roles=["バリスタ", "ホール", "キッチン", "店長"],
         out=["food", "service"]),
    dict(id="co_ca_06", name="(株)公園通りロースタリー", key="CA", detail="焙煎+カフェ",
         products=["コーヒー豆の焙煎・卸", "ロースタリーカフェ運営"],
         emp=19, band="10-49", roles=["バリスタ", "ホール", "キッチン", "店長"],
         out=["food", "goods"]),
    # --- 生活関連サービス/美容 4社(shop) ---
    dict(id="co_bt_01", name="(株)渋谷ヘアデザイン", key="BT", detail="美容室",
         products=["ヘアカット・カラー・パーマ等の美容サービス"],
         emp=24, band="10-49", roles=["スタイリスト", "アシスタント", "受付", "店長"],
         out=["service"]),
    dict(id="co_bt_02", name="(株)センガイ・ビューティ", key="BT", detail="ヘア/ネイルサロン",
         products=["ヘア・ネイルのトータルサロン運営"],
         emp=18, band="10-49", roles=["スタイリスト", "アシスタント", "受付", "店長"],
         out=["service"]),
    dict(id="co_bt_03", name="(株)公園通りサロン", key="BT", detail="トータルビューティ",
         products=["ヘア・エステ・メイクのビューティサービス"],
         emp=16, band="10-49", roles=["スタイリスト", "アシスタント", "受付", "店長"],
         out=["service"]),
    dict(id="co_bt_04", name="(株)宮益スパ&リラクゼーション", key="BT", detail="リラク/エステ",
         products=["リラクゼーション・エステの提供"],
         emp=14, band="10-49", roles=["スタイリスト", "アシスタント", "受付", "店長"],
         out=["service"]),
    # --- 不動産 3社(office) ---
    dict(id="co_re_01", name="(株)渋谷不動産パートナーズ", key="RE", detail="賃貸仲介/管理",
         products=["賃貸物件の仲介・管理"],
         emp=30, band="10-49", roles=["営業", "事務", "物件管理"],
         out=["service"]),
    dict(id="co_re_02", name="(株)道玄坂リアルエステート", key="RE", detail="オフィス仲介",
         products=["オフィス・店舗物件の仲介"],
         emp=22, band="10-49", roles=["営業", "事務", "物件管理"],
         out=["service"]),
    dict(id="co_re_03", name="(株)宮下地所", key="RE", detail="開発/PM",
         products=["不動産開発・プロパティマネジメント"],
         emp=55, band="50-99", roles=["営業", "事務", "物件管理"],
         out=["service"]),
    # --- 専門サービス 3社(office) ---
    dict(id="co_pr_01", name="スクランブル総合会計事務所", key="PR", detail="税務/会計",
         products=["税務・会計・記帳代行"],
         emp=26, band="10-49", roles=["会計士", "コンサルタント", "営業", "アシスタント"],
         out=["service"]),
    dict(id="co_pr_02", name="(株)渋谷ヒューマンリソース", key="PR", detail="人材紹介/派遣",
         products=["人材紹介・人材派遣"],
         emp=38, band="10-49", roles=["コンサルタント", "会計士", "営業", "アシスタント"],
         out=["service"]),
    dict(id="co_pr_03", name="(株)センガイ・コンサルティング", key="PR", detail="経営/ITコンサル",
         products=["経営・IT導入コンサルティング"],
         emp=34, band="10-49", roles=["コンサルタント", "会計士", "営業", "アシスタント"],
         out=["service"]),
    # --- 卸売商社/メーカー本社 2社(office) ---
    dict(id="co_wh_01", name="(株)渋谷トレーディング", key="WH", detail="生活雑貨卸",
         products=["生活雑貨の卸売・輸入"],
         emp=48, band="10-49", roles=["営業", "企画", "物流管理", "事務"],
         out=["goods", "service"]),
    dict(id="co_wh_02", name="(株)ハチ公プロダクツ", key="WH", detail="雑貨メーカー本社",
         products=["雑貨の企画・製造(本社機能)"],
         emp=52, band="50-99", roles=["企画", "営業", "物流管理", "事務"],
         out=["goods"]),
    # --- 運輸・物流/配達プラットフォーム 1社(office拠点 + 配達員はギグ=固定拠点なし) ---
    dict(id="co_lg_01", name="(株)スクランブル・デリバリー", key="LG", detail="配達プラットフォーム",
         products=["フードデリバリー等の配達プラットフォーム運営", "ラストワンマイル配送"],
         emp=80, band="50-99", roles=["オペレーション", "営業", "配達員"],
         out=["service"]),
    # --- フリーランス/個人事業 umbrella 3社(POI束縛なし。自営) ---
    dict(id="co_fl_01", name="渋谷フリーランス・コレクティブ", key="GF", detail="制作系フリーランス",
         products=["デザイン・開発・ライティングの個人事業"],
         emp=1, band="1", roles=["フリーランス"], out=["service", "content"], poi_cat=None,
         wage_tier="自営"),
    dict(id="co_fl_02", name="渋谷フォトグラファーズ(個人事業)", key="GP", detail="写真/映像",
         products=["写真・映像撮影の個人事業"],
         emp=1, band="1", roles=["フォトグラファー"], out=["content"], poi_cat=None,
         wage_tier="自営"),
    dict(id="co_mu_01", name="渋谷ミュージシャンズ・ギルド", key="GM", detail="音楽/ライブ",
         products=["音楽・ライブ活動(個人)"],
         emp=1, band="1", roles=["ミュージシャン"], out=["content"], poi_cat=None,
         wage_tier="自営"),
]

# 業種キー → (industry 表示名, POI カテゴリ, wage_tier, 会社員が就く generalist 職種)
INDUSTRY = {
    "IT": ("情報通信業", "office", "会社員", ["営業", "プロダクトマネージャー", "コーポレート"]),
    "AD": ("広告・コンテンツ制作", "office", "会社員", ["プランナー", "ディレクター", "営業", "制作進行"]),
    "AP": ("卸売・小売業(アパレル)", "shop", "店員", ["店長", "バイヤー", "EC運営", "VMD"]),
    "CA": ("飲食・カフェ", "food", "店員", ["店長"]),
    "BT": ("生活関連サービス(美容)", "shop", "店員", ["店長", "受付"]),
    "RE": ("不動産業", "office", "会社員", ["営業", "事務", "物件管理"]),
    "PR": ("専門サービス業", "office", "会社員", ["コンサルタント", "会計士", "営業", "アシスタント"]),
    "WH": ("卸売業/メーカー本社", "office", "会社員", ["営業", "企画", "物流管理", "事務"]),
    "LG": ("運輸・物流業", "office", "会社員", ["オペレーション", "営業"]),
    "GF": ("個人事業(制作)", None, "自営", ["フリーランス"]),
    "GP": ("個人事業(写真)", None, "自営", ["フォトグラファー"]),
    "GM": ("個人事業(音楽)", None, "自営", ["ミュージシャン"]),
}

# 学校(架空名。区立の実数 = 小17/中7/一貫1 は §2.1。台帳は代表校を各種そろえる)。
SCHOOLS: list[dict] = [
    dict(id="sc_es_01", name="渋谷区立スクランブル小学校", type="区立小学校",
         subjects=["国語", "社会", "算数", "理科", "生活", "音楽", "図画工作", "家庭", "体育",
                   "外国語", "道徳", "外国語活動", "総合的な学習の時間", "特別活動"],
         period_min=45, periods_per_day=5, start="08:30", capacity=620),
    dict(id="sc_es_02", name="渋谷区立宮益小学校", type="区立小学校",
         subjects=["国語", "社会", "算数", "理科", "生活", "音楽", "図画工作", "家庭", "体育",
                   "外国語", "道徳", "外国語活動", "総合的な学習の時間", "特別活動"],
         period_min=45, periods_per_day=5, start="08:30", capacity=540),
    dict(id="sc_js_01", name="渋谷区立道玄坂中学校", type="区立中学校",
         subjects=["国語", "社会", "数学", "理科", "音楽", "美術", "保健体育", "技術・家庭",
                   "外国語", "道徳", "総合的な学習の時間", "特別活動"],
         period_min=50, periods_per_day=6, start="08:30", capacity=480),
    dict(id="sc_js_02", name="渋谷区立神南中学校", type="区立中学校",
         subjects=["国語", "社会", "数学", "理科", "音楽", "美術", "保健体育", "技術・家庭",
                   "外国語", "道徳", "総合的な学習の時間", "特別活動"],
         period_min=50, periods_per_day=6, start="08:30", capacity=430),
    dict(id="sc_ce_01", name="渋谷区立公園通り小中一貫校", type="小中一貫校",
         subjects=["国語", "社会", "算数・数学", "理科", "音楽", "美術・図画工作", "保健体育",
                   "技術・家庭", "外国語", "道徳", "総合的な学習の時間", "特別活動"],
         period_min=50, periods_per_day=6, start="08:30", capacity=820),
    dict(id="sc_hs_01", name="渋谷セントラル高等学校(仮称)", type="高校",
         subjects=["国語", "地理歴史", "公民", "数学", "理科", "保健体育", "芸術", "外国語",
                   "家庭", "情報", "総合的な探究の時間", "特別活動"],
         period_min=50, periods_per_day=6, start="08:40", capacity=900),
    dict(id="sc_un_01", name="渋谷総合大学(仮称)", type="大学",
         faculties=["文学部", "経済学部", "社会学部", "情報学部", "芸術学部"],
         lecture_fields=["文学・言語", "経済学", "社会学", "情報科学・プログラミング", "メディア表現"],
         period_min=90, periods_per_day=5, start="09:00", capacity=6000),
    dict(id="sc_un_02", name="宮益工科大学(仮称)", type="大学",
         faculties=["理工学部", "情報学部", "建築学部", "デザイン工学部"],
         lecture_fields=["数理・物理", "情報科学・AI", "建築・都市", "デザイン工学"],
         period_min=90, periods_per_day=5, start="09:00", capacity=4200),
    dict(id="sc_vo_01", name="渋谷デザイン&ファッション専門学校(仮称)", type="専門学校",
         lecture_fields=["グラフィックデザイン", "ファッションデザイン", "映像制作", "インテリア"],
         period_min=90, periods_per_day=4, start="09:20", capacity=1200),
    dict(id="sc_vo_02", name="渋谷IT&ゲーム専門学校(仮称)", type="専門学校",
         lecture_fields=["プログラミング", "ゲーム開発", "AI・データサイエンス", "Web/UXデザイン"],
         period_min=90, periods_per_day=4, start="09:20", capacity=1500),
]

# 就業時間の標準(§3): 会社は 9:00-18:00(実働8h+休憩1h)。
WORK_HOURS = {"start": "09:00", "end": "18:00", "break_min": 60, "work_min": 480}

# 職業 → 就く業種キー候補(会社員は決定論ハッシュで業種内の1社へ)
OCC_INDUSTRIES: dict[str, list[str]] = {
    "会社員": ["IT", "AD", "RE", "PR", "WH", "LG"],
    "エンジニア": ["IT"],
    "デザイナー": ["IT", "AD"],
    "アパレル店員": ["AP"],
    "カフェ店員": ["CA"],
    "美容師": ["BT"],
    "フリーランス": ["GF"],
    "写真家": ["GP"],
    "バンドマン": ["GM"],
    # 大学生 は学校へ(下記 _assign_student)。無職 は未配属。
}

# 職業 → specialist 職種(会社員は generalist を業種から選ぶので除外)
OCC_SPECIALIST: dict[str, str] = {
    "エンジニア": "エンジニア",
    "デザイナー": "デザイナー",
    "アパレル店員": "販売スタッフ",
    "美容師": "スタイリスト",
    "配達員": "配達員",
    "フリーランス": "フリーランス",
    "写真家": "フォトグラファー",
    "バンドマン": "ミュージシャン",
}
# カフェ店員は現場職を hash で振り分け
CAFE_ROLES = ["バリスタ", "ホール", "キッチン"]


# ----------------------------------------------------------------------------
# 地図 POI の読み込みと束縛
# ----------------------------------------------------------------------------
def load_map(path: Path) -> tuple[dict[str, list[dict]], set[str]]:
    """POI をカテゴリ別に(id 昇順で)集める + 全ノード集合を返す。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    by_cat: dict[str, list[dict]] = {}
    for p in d.get("pois", []):
        by_cat.setdefault(p["cat"], []).append(p)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda p: p["id"])
    nodes = {n["id"] if isinstance(n, dict) else n for n in d.get("nodes", [])}
    return by_cat, nodes


def _distinct_by_node(pool: list[dict]) -> list[dict]:
    """同一 node に複数 POI がある地図なので、node ごとに id 最小の1件へ間引く(束縛を空間的に分散)。"""
    seen, out = set(), []
    for p in pool:                                   # pool は id 昇順
        if p["node"] not in seen:
            seen.add(p["node"])
            out.append(p)
    return out


def _poi_ref(pool: list[dict], idx: int) -> dict | None:
    """distinct-node pool から idx 番目(なければ wrap)の POI を node/座標参照で返す。実在店名は載せない。"""
    if not pool:
        return None
    p = pool[idx % len(pool)]
    return {"poi_id": p["id"], "node": p["node"],
            "x": round(float(p["x"]), 1), "y": round(float(p["y"]), 1)}


def build_orgs(by_cat: dict[str, list[dict]], seed: int) -> dict:
    """架空の組織台帳を組み立て、各組織を実在 POI(node)へ決定論束縛する。"""
    distinct: dict[str, list[dict]] = {c: _distinct_by_node(v) for c, v in by_cat.items()}
    used_by_cat: dict[str, int] = {}
    companies = []
    for c in COMPANIES:
        ind = INDUSTRY[c["key"]]
        industry_name, default_cat, default_tier, generalist = ind
        poi_cat = c.get("poi_cat", default_cat)
        wage_tier = c.get("wage_tier", default_tier)
        poi = None
        if poi_cat is not None:
            idx = used_by_cat.get(poi_cat, 0)
            poi = _poi_ref(distinct.get(poi_cat, []), idx)
            used_by_cat[poi_cat] = idx + 1
        companies.append({
            "id": c["id"], "name": c["name"], "industry": industry_name,
            "industry_key": c["key"],
            "sector_detail": c["detail"], "products_services": c["products"],
            "size": {"employees": c["emp"], "band": c["band"]},
            "wage_tier": wage_tier,
            "workplace_poi": ({"cat": poi_cat, **poi} if poi else {"cat": poi_cat}),
            "roles": c["roles"], "generalist_roles": generalist,
            "output_kinds": c["out"],
        })
    schools = []
    for s in SCHOOLS:
        idx = used_by_cat.get("__school__", 0)
        pool = distinct.get("education") or distinct.get("school") or []
        poi = _poi_ref(pool, idx)
        used_by_cat["__school__"] = idx + 1
        rec = {
            "id": s["id"], "name": s["name"], "school_type": s["type"],
            "timetable": {"period_min": s["period_min"],
                          "periods_per_day": s["periods_per_day"],
                          "start": s["start"]},
            "capacity": s["capacity"],
            "workplace_poi": ({"cat": "education", **poi} if poi else {"cat": "education"}),
            "roles": ["教員", "職員", "学生"],
        }
        if "subjects" in s:
            rec["subjects"] = s["subjects"]
        if "faculties" in s:
            rec["faculties"] = s["faculties"]
        if "lecture_fields" in s:
            rec["lecture_fields"] = s["lecture_fields"]
        schools.append(rec)
    return {
        "meta": {
            "generator": "scripts/build_orgs.py", "seed": seed,
            "note": ("架空の組織台帳(R17: 実在企業名・学校名なし)。構成比・事業内容・"
                     "教育課程・時間構造は現実準拠。設計根拠と出典は "
                     "docs/research/shibuya-organizations.md。POI束縛は地図ノード参照のみ。"),
            "sources_doc": "docs/research/shibuya-organizations.md",
            "industry_share_reference": {
                "サービス業": 0.29, "卸売・小売業": 0.28, "飲食店・宿泊業": 0.15,
                "不動産業": 0.09, "情報通信業": 0.08, "製造業": 0.035, "建設業": 0.03,
                "その他(医療福祉教育金融運輸)": 0.10,
                "note": "東京商工会議所渋谷支部(事業所統計・平成18/総数30,572)。IT は渋谷特有の集積で台帳は現実比よりやや厚め。"},
            "work_hours_standard": WORK_HOURS,
            "school_ladder_shibuya": {"区立小学校": 17, "区立中学校": 7, "小中一貫校": 1,
                                      "note": "区の実数(§2.1)。台帳は各段階の代表校を収録。"},
            "counts": {"companies": len(companies), "schools": len(schools)},
        },
        "companies": companies,
        "schools": schools,
    }


# ----------------------------------------------------------------------------
# 配属(エージェント → 組織)
# ----------------------------------------------------------------------------
def _industry_pool(orgs: dict, keys: list[str]) -> list[dict]:
    keyset = set(keys)
    return [c for c in orgs["companies"] if c.get("industry_key") in keyset]


def _school_pool(orgs: dict, types: set[str]) -> list[dict]:
    return [s for s in orgs["schools"] if s["school_type"] in types]


def _age_stage(age: int) -> set[str]:
    """年齢 → 学校段階(将来の低年齢名簿にも対応。現行名簿は全員18+=大学/専門)。"""
    if age < 12:
        return {"区立小学校", "小中一貫校"}
    if age < 15:
        return {"区立中学校", "小中一貫校"}
    if age < 18:
        return {"高校"}
    return {"大学", "専門学校"}


def _resident_type(p: dict) -> str:
    if not p.get("visitor"):
        return "resident"
    return "commuter" if p.get("commute") else "leisure_visitor"


def assign_agent(idx: int, p: dict, orgs: dict, tag: str) -> dict | None:
    """1エージェントを決定論的に組織へ配属。無職/学生外の未対応職業は None。"""
    occ = p["occupation"]
    name = p.get("name", "")
    if occ == "無職":
        return None
    if occ == "大学生":
        pool = _school_pool(orgs, _age_stage(int(p.get("age", 20))))
        if not pool:
            return None
        org = _pick(pool, tag, idx, name, "school")
        return {"org_id": org["id"], "role": "学生"}
    # 配達員(ギグ): co_lg_01 の配達員として membership(POI束縛なし)
    if occ == "配達員":
        return {"org_id": "co_lg_01", "role": "配達員"}
    keys = OCC_INDUSTRIES.get(occ)
    if not keys:
        return None
    pool = _industry_pool(orgs, keys)
    if not pool:
        return None
    org = _pick(pool, tag, idx, name, occ)
    if occ == "会社員":
        role = _pick(org["generalist_roles"], tag, idx, name, "role")
    elif occ == "カフェ店員":
        role = _pick(CAFE_ROLES, tag, idx, name, "role")
    else:
        role = OCC_SPECIALIST.get(occ, occ)
    return {"org_id": org["id"], "role": role}


def build_assignments(orgs: dict, roster_path: Path, tag: str,
                      orgs_file: str) -> dict:
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    personas = roster["personas"] if isinstance(roster, dict) else roster
    org_by_id = {c["id"]: c for c in orgs["companies"]}
    org_by_id.update({s["id"]: s for s in orgs["schools"]})
    assignments, unassigned = [], []
    by_reason = {"assigned": 0, "unemployed": 0, "no_rule": 0}
    by_kind = {"company": 0, "school": 0}
    by_restype_assigned = {"resident": 0, "commuter": 0, "leisure_visitor": 0}
    for idx, p in enumerate(personas):
        occ = p["occupation"]
        rt = _resident_type(p)
        a = assign_agent(idx, p, orgs, tag)
        if a is None:
            reason = "unemployed" if occ == "無職" else "no_rule"
            by_reason[reason] += 1
            unassigned.append({"agent_index": idx, "name": p.get("name", ""),
                               "occupation": occ, "resident_type": rt,
                               "reason": reason})
            continue
        by_reason["assigned"] += 1
        by_restype_assigned[rt] += 1
        by_kind["school" if a["org_id"].startswith("sc_") else "company"] += 1
        assignments.append({"agent_index": idx, "org_id": a["org_id"],
                            "role": a["role"], "occupation": occ,
                            "resident_type": rt,
                            "org_name": org_by_id[a["org_id"]]["name"]})
    n = len(personas)
    return {
        "meta": {
            "roster": str(roster_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "tag": tag, "orgs_file": orgs_file, "n_agents": n,
            "n_assigned": len(assignments),
            "coverage": round(len(assignments) / n, 3) if n else 0.0,
            "by_reason": by_reason, "by_org_kind": by_kind,
            "assigned_by_resident_type": by_restype_assigned,
            "note": ("固定拠点をもつ職業(会社員/エンジニア/デザイナー/各店員/大学生)は POI束縛の"
                     "会社・学校へ、自営(フリーランス/写真家/バンドマン/配達員)は POI束縛なしの"
                     "個人事業umbrellaへ配属。無職は経済的所属なしで未配属。来街者(leisure_visitor)も"
                     "エンジンが職場POIを与える職業なら配属する(assigned_by_resident_type で内訳確認可)。"),
        },
        "unassigned": unassigned,
        "assignments": assignments,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="架空の組織台帳と配属を生成(バッチE)")
    ap.add_argument("--map", default=str(DEFAULT_MAP), help="地図 JSON(既定 shibuya_osm.json)")
    ap.add_argument("--roster", default=None,
                    help="1名簿だけ再生成(既定は既知2名簿すべて)")
    ap.add_argument("--out", default=None, help="配属の出力先(--roster と併用)")
    ap.add_argument("--orgs-out", default=None,
                    help="台帳の出力先(既定 data/organizations_shibuya.json。広域地図用に"
                         "別ファイルへ出すときに指定=正準台帳を上書きしない)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    map_path = Path(args.map)
    if not map_path.is_absolute():
        map_path = REPO_ROOT / map_path
    by_cat, nodes = load_map(map_path)
    orgs = build_orgs(by_cat, args.seed)
    orgs["meta"]["map"] = str(map_path.relative_to(REPO_ROOT)).replace("\\", "/")
    orgs_out = (Path(args.orgs_out) if args.orgs_out
                else DATA / "organizations_shibuya.json")
    if not orgs_out.is_absolute():
        orgs_out = REPO_ROOT / orgs_out
    orgs_out.write_text(json.dumps(orgs, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"written: {orgs_out}  (会社 {len(orgs['companies'])} / 学校 {len(orgs['schools'])})")

    rosters = [Path(args.roster)] if args.roster else list(DEFAULT_ROSTERS)
    orgs_file = str(orgs_out.relative_to(REPO_ROOT)).replace("\\", "/")
    for rp in rosters:
        if not rp.is_absolute():
            rp = REPO_ROOT / rp
        tag = rp.stem.replace("personas_", "")
        out = (Path(args.out) if (args.out and args.roster)
               else DATA / f"org_assignments_{tag}.json")
        if not out.is_absolute():
            out = REPO_ROOT / out
        result = build_assignments(orgs, rp, tag, orgs_file)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        m = result["meta"]
        print(f"written: {out}  (配属 {m['n_assigned']}/{m['n_agents']} = "
              f"{m['coverage']:.0%}, 未配属 {len(result['unassigned'])})")


if __name__ == "__main__":
    main()
