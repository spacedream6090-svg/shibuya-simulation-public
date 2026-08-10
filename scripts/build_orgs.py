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
import random
import sys
from pathlib import Path

# cp932 コンソール(Windows 既定)で Unicode を print しても死なない(devlog-protocol の掟)。
try:  # pragma: no cover - 環境依存
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
DEFAULT_MAP = DATA / "shibuya_osm.json"
DEFAULT_WIDE_MAP = DATA / "shibuya_osm_wide_v7.json"      # 分布駆動(--dist)の既定広域地図
DEFAULT_DIST_OUT = DATA / "organizations_shibuya_wide11k.json"  # 分布駆動の既定出力(新ファイル)
# センサス較正(--census)の既定出力。正準 11k 台帳を事故で上書きしないよう別ファイルにする。
DEFAULT_CENSUS_OUT = DATA / "organizations_shibuya_census.json"
DEFAULT_CENSUS_JSON = DATA / "realworld" / "census_r3" / "station_area_industry.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "research" / "org-book-11k.md"
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


# ============================================================================
# 分布駆動モード(P6: 台帳 52 → ~1.1万。経済センサス分布からの手続き生成)
# ----------------------------------------------------------------------------
# 目的(docs/plans/w2-execution-plan.md §4 P6): 従業者ペルソナ生成(P5)の需要源となる
# ~1.1万組織を、渋谷区の産業大分類×従業者規模帯の分布から seed 付き決定論で手続き生成する。
#
# 正直な近似の記録(捏造しない=repo の「実数と偽装しない」掟):
#   - 産業大分類の *事業所数* 構成比: 東京商工会議所渋谷支部が引く事業所統計(平成18・総数30,572)
#     の業種別内訳(docs/research/shibuya-organizations.md §1.1)を骨格に採用。IT はビットバレー
#     集積を反映してやや厚め。
#   - *規模帯*(従業者規模8区分)の分布: 渋谷区の産業大分類×従業者規模の実クロス表
#     (e-Stat 経済センサス stat_infid=000031720779)は JavaScript 描画で静的抽出できず
#     (shibuya-organizations.md §1.4 / shibuya-population.md §9 に記録済み)。よって
#     「都市部事業所は小規模多数+少数の中〜大規模」という一般構造(全国経済センサスの裾長分布)を
#     ベースに、渋谷 bbox の実測アンカー(§5: 事業所~1.19万・従業者~25.7万 → 平均~21.6人/所)へ
#     合うよう裾を補正した *近似分布* を用いる。産業別の規模傾斜(オフィス系=大・小売飲食=小)も
#     この一般構造に基づく近似であり、実クロス表の再現を主張しない。
#   - *硬いアンカー*(実数): 事業所数 ~1.1万(目標 count)と従業者総和 ~25.7万
#     (shibuya-population.md §5・町丁目合算=信頼度高)には ±10% で整合させる。
#   - 産業別の *従業者シェア* は生成の副産物として meta に記録するが、区の実クロス表からの抽出値
#     ではない(近似)。令和3年経済センサスの区従業者(IT 106,574 / 卸小売 108,823 等・§5)とは
#     オーダーで整合するが厳密一致は主張しない。
#
# 決定論: (map, seed, count) から random.Random(seed) の固定消費順で一意に決まる。LLM 非使用。
# ============================================================================

# 従業者規模帯(経済センサス 8 区分)。(表示 label, lo, hi)。300+ は上限を 1200 で近似打切り。
SIZE_BANDS: list[tuple[str, int, int]] = [
    ("1-4", 1, 4), ("5-9", 5, 9), ("10-19", 10, 19), ("20-29", 20, 29),
    ("30-49", 30, 49), ("50-99", 50, 99), ("100-299", 100, 299), ("300+", 300, 1200),
]

# 規模クラス別の規模帯構成(事業所数シェア=正規化前。裾長。都市部の一般構造+渋谷補正)。
#   SMALL  平均 ~11 人 / RETAIL ~20 / MED ~26 / LARGE ~50。加重平均 ~23.4 → 総従業者 ~25.7万。
SIZE_CLASS_DIST: dict[str, list[float]] = {
    "SMALL":  [0.55, 0.22, 0.12, 0.040, 0.030, 0.020, 0.008, 0.002],
    "RETAIL": [0.47, 0.22, 0.15, 0.055, 0.045, 0.030, 0.022, 0.008],
    "MED":    [0.38, 0.20, 0.16, 0.075, 0.065, 0.050, 0.038, 0.008],
    "LARGE":  [0.24, 0.18, 0.16, 0.090, 0.080, 0.065, 0.062, 0.030],
}

# 産業大分類。share=事業所数構成比(正規化前)/ cls=規模クラス / cat=職場POIカテゴリ / tier=賃金区分。
INDUSTRY_SPEC: list[dict] = [
    dict(key="IT", name="情報通信業", cat="office", tier="会社員", cls="LARGE", share=0.08,
         details=["業務SaaS", "受託開発/DX", "ゲーム開発", "クラウド/API", "データ分析/AI", "Webメディア"],
         products=["業務SaaSの開発提供", "Web・業務システムの受託開発", "アプリ/ゲームの開発運営"],
         roles=["エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"],
         generalist=["営業", "プロダクトマネージャー", "コーポレート"],
         output=["software", "service"],
         words=["システムズ", "ソフトウェア", "テック", "デジタル", "ラボ", "データワークス", "クラウド", "ネットワークス"]),
    dict(key="WR", name="卸売業・小売業", cat="shop", tier="店員", cls="RETAIL", share=0.28,
         details=["アパレル小売", "雑貨・ライフスタイル", "セレクトショップ", "コスメ物販", "食品小売", "生活雑貨卸"],
         products=["衣料・服飾雑貨の小売", "生活雑貨・日用品の販売", "商品の卸売・仕入"],
         roles=["販売スタッフ", "店長", "バイヤー", "EC運営", "VMD", "在庫管理"],
         generalist=["店長", "バイヤー", "EC運営", "VMD"],
         output=["goods", "retail"],
         words=["商会", "物産", "ストア", "マーケット", "トレーディング", "リテール", "商店", "セレクト"]),
    dict(key="FB", name="宿泊業・飲食サービス業", cat="food", tier="店員", cls="SMALL", share=0.15,
         details=["カフェ", "ダイニング", "ベーカリー", "定食・ランチ", "バー", "テイクアウト"],
         products=["飲食の提供・店舗運営", "テイクアウト・デリバリーの提供"],
         roles=["バリスタ", "ホール", "キッチン", "店長"],
         generalist=["店長"],
         output=["food", "service"],
         words=["フーズ", "ダイニング", "キッチン", "珈琲", "食堂", "テーブル", "ブリュワリー", "キュイジーヌ"]),
    dict(key="PS", name="学術研究・専門・技術サービス業", cat="office", tier="会社員", cls="MED", share=0.10,
         details=["広告運用/制作", "経営/ITコンサル", "税務・会計", "デザイン/ブランディング", "法務", "人材紹介"],
         products=["広告・クリエイティブの制作運用", "経営・IT・専門コンサルティング", "会計・法務・人材の専門サービス"],
         roles=["プランナー", "ディレクター", "コンサルタント", "デザイナー", "営業", "制作進行"],
         generalist=["プランナー", "ディレクター", "営業", "制作進行"],
         output=["ad", "content", "service"],
         words=["コンサルティング", "総研", "パートナーズ", "デザイン", "クリエイティブ", "アドワークス", "法務事務所", "会計事務所"]),
    dict(key="LS", name="生活関連サービス業・娯楽業", cat="shop", tier="店員", cls="SMALL", share=0.07,
         details=["美容室", "ネイル/エステ", "リラクゼーション", "アミューズメント", "フィットネス", "写真スタジオ"],
         products=["ヘア・美容・リラクゼーションの提供", "娯楽・レジャーの提供"],
         roles=["スタイリスト", "アシスタント", "受付", "店長"],
         generalist=["店長", "受付"],
         output=["service"],
         words=["ビューティ", "サロン", "スパ", "ウェルネス", "スタジオ", "リラクゼーション", "ヘアデザイン", "エンタメ"]),
    dict(key="RE", name="不動産業・物品賃貸業", cat="office", tier="会社員", cls="MED", share=0.09,
         details=["賃貸仲介/管理", "オフィス仲介", "開発/PM", "物品賃貸"],
         products=["不動産の仲介・管理", "不動産開発・プロパティマネジメント"],
         roles=["営業", "事務", "物件管理"],
         generalist=["営業", "事務", "物件管理"],
         output=["service"],
         words=["地所", "不動産", "エステート", "リアルティ", "プロパティ", "開発", "住宅", "レンタル"]),
    dict(key="SV", name="サービス業(他に分類されないもの)", cat="service", tier="会社員", cls="MED", share=0.06,
         details=["人材派遣", "施設管理/メンテ", "BPO/事務代行", "警備", "各種代行"],
         products=["人材派遣・業務請負", "施設管理・メンテナンス・各種代行サービス"],
         roles=["オペレーション", "営業", "スタッフ", "事務"],
         generalist=["オペレーション", "営業", "事務"],
         output=["service"],
         words=["サービス", "サポート", "ワークス", "エージェンシー", "ソリューションズ", "メンテナンス", "ヒューマンリソース"]),
    dict(key="MW", name="医療・福祉", cat="service", tier="会社員", cls="MED", share=0.04,
         details=["クリニック", "歯科", "調剤薬局", "介護/福祉", "治療院"],
         products=["医療・診療サービス", "介護・福祉サービス"],
         roles=["医療スタッフ", "受付", "事務", "介護スタッフ"],
         generalist=["受付", "事務"],
         output=["service"],
         words=["クリニック", "ケア", "メディカル", "ヘルスケア", "福祉サービス", "治療院", "調剤"]),
    dict(key="ED", name="教育・学習支援業", cat="service", tier="会社員", cls="MED", share=0.03,
         details=["学習塾", "英会話", "音楽教室", "資格スクール", "予備校"],
         products=["学習指導・教育サービス", "各種スクールの運営"],
         roles=["講師", "事務", "教室長"],
         generalist=["事務", "教室長"],
         output=["service", "content"],
         words=["ゼミナール", "アカデミー", "スクール", "学習塾", "教育", "カレッジ", "エデュ"]),
    dict(key="FI", name="金融業・保険業", cat="office", tier="会社員", cls="LARGE", share=0.02,
         details=["投資/ファンド", "保険代理", "証券", "決済/フィンテック"],
         products=["金融・投資サービス", "保険・証券の提供"],
         roles=["アナリスト", "営業", "事務", "コンサルタント"],
         generalist=["営業", "事務", "コンサルタント"],
         output=["service"],
         words=["キャピタル", "フィナンシャル", "ファンド", "保険サービス", "証券", "インベストメント"]),
    dict(key="CN", name="建設業", cat="office", tier="会社員", cls="MED", share=0.03,
         details=["建築設計/施工", "内装/リフォーム", "設備工事", "土木"],
         products=["建築・内装の設計施工", "設備・リフォーム工事"],
         roles=["施工管理", "設計", "営業", "事務"],
         generalist=["営業", "事務"],
         output=["service"],
         words=["建設", "工務店", "建築", "エンジニアリング", "コンストラクション", "設備"]),
    dict(key="MF", name="製造業", cat="office", tier="会社員", cls="MED", share=0.03,
         details=["雑貨企画/本社", "食品加工", "アパレル生産管理", "印刷"],
         products=["製品の企画・製造(本社機能)", "受託製造・加工"],
         roles=["企画", "生産管理", "営業", "事務"],
         generalist=["企画", "営業", "事務"],
         output=["goods", "service"],
         words=["製作所", "プロダクツ", "工業", "ファクトリー", "メーカーズ", "製造"]),
    dict(key="TR", name="運輸業・郵便業", cat="office", tier="会社員", cls="MED", share=0.015,
         details=["配送/物流", "倉庫", "運送", "宅配拠点"],
         products=["配送・物流サービス", "倉庫・輸送の受託"],
         roles=["ドライバー", "オペレーション", "倉庫管理", "事務"],
         generalist=["オペレーション", "事務"],
         output=["service"],
         words=["運輸", "物流", "ロジスティクス", "エクスプレス", "トランスポート", "配送"]),
    dict(key="CS", name="複合サービス事業・その他", cat="service", tier="会社員", cls="SMALL", share=0.005,
         details=["協同組合", "郵便局窓口", "複合サービス"],
         products=["複合サービスの提供", "組合・共同事業の運営"],
         roles=["スタッフ", "事務", "窓口"],
         generalist=["事務", "窓口"],
         output=["service"],
         words=["サービス", "協同", "コンプレックス", "ユニオン", "エージェント"]),
]
# ============================================================================
# センサス較正モード専用の追加業種(Wave 4 IV-1。**--census を渡したときだけ台帳に載る**)
# ----------------------------------------------------------------------------
# 監査の指摘: 接客(work.serve_by_cat)は {food, cafe, nightlife, shop} を張っているのに、
# 台帳の職場カテゴリは {office, shop, food, service} しか無い = **nightlife の職場が
# 構造的に存在せず、ナイトライフ消費は永遠に無人接客(unstaffed)になる**。
#
# 令和3年経済センサスの町丁目表では、駅周辺13町丁目に中分類 80「娯楽業」が実在する
# (254 事業所 / 4,980 人 = 事業所の 2.6%)。既存 LS(生活関連サービス業・娯楽業)は
# 大分類 N をまるごと1バケットにして cat="shop" に潰していたため、この 254 事業所は
# 「服屋」として台帳に載っていた。--census ではセンサス中分類にならって
#   N78 洗濯・理容・美容・浴場業 + N79 その他の生活関連サービス業 → LS(cat=service)
#   N80 娯楽業                                                   → AM(cat=nightlife)
# の 2 バケットへ割る。★INDUSTRY_SPEC 本体には足さない(足すと既定分布が動く)。
CENSUS_EXTRA_INDUSTRY_SPEC: list[dict] = [
    dict(key="AM", name="娯楽業", cat="nightlife", tier="店員", cls="SMALL", share=0.0,
         details=["ライブハウス", "クラブ/バー併設", "カラオケ", "ゲームセンター", "劇場・興行", "スポーツ施設"],
         products=["音楽・興行・遊興の提供", "深夜帯の娯楽サービス運営"],
         roles=["フロア", "バーテンダー", "音響", "受付", "店長"],
         generalist=["店長", "受付"],
         output=["service", "content"],
         words=["ホール", "ラウンジ", "ナイト", "サウンド", "ステージ", "アミューズメント", "クラブ", "シアター"]),
]

# --census のときだけ差し替える業種定義(既定モードの値は 1 バイトも変えない)。
#   LS: 娯楽業を AM へ切り出したので、名称・事業内容から娯楽を外す。職場カテゴリは
#       「物を売る店(shop)」ではなく役務提供(service)= services.py の受給側 POI カテゴリと揃え、
#       需要(service_use)と供給(serve)が同じ接点で対になるようにする。
#   ★details の要素数は既定と同じに保つ(rng の消費数を業種間で揃えるため)。
CENSUS_SPEC_OVERRIDE: dict[str, dict] = {
    "LS": {"name": "生活関連サービス業", "cat": "service",
           "details": ["美容室", "ネイル/エステ", "リラクゼーション", "クリーニング", "フィットネス", "写真スタジオ"],
           "products": ["ヘア・美容・リラクゼーションの提供", "洗濯・理容・浴場等の生活関連サービスの提供"]},
}
CENSUS_CAT_OVERRIDE: dict[str, str] = {k: v["cat"] for k, v in CENSUS_SPEC_OVERRIDE.items()
                                       if "cat" in v}


def _census_spec(it: dict, census_mode: bool) -> dict:
    """--census のときだけ業種定義を差し替えた辞書を返す。既定モードは同一オブジェクト(不変)。"""
    ov = CENSUS_SPEC_OVERRIDE.get(it["key"]) if census_mode else None
    return {**it, **ov} if ov else it

ALL_INDUSTRY_SPEC: list[dict] = INDUSTRY_SPEC + CENSUS_EXTRA_INDUSTRY_SPEC
IND_BY_KEY: dict[str, dict] = {it["key"]: it for it in ALL_INDUSTRY_SPEC}

# 業種別の営業時間・シフト(shift_pattern)。オフィス=平日日勤 / 小売飲食=長時間・週末含む・交代制。
# nightlife は --census で生える AM 専用(close < open = 日跨ぎ。既定モードでは誰も引かない)。
SHIFT_BY_CAT: dict[str, dict] = {
    "office":  {"open": "09:00", "close": "18:00", "days": "mon-fri", "shift_hours": 8, "rotates": False},
    "shop":    {"open": "10:00", "close": "21:00", "days": "all",     "shift_hours": 8, "rotates": True},
    "food":    {"open": "07:00", "close": "23:00", "days": "all",     "shift_hours": 8, "rotates": True},
    "service": {"open": "09:00", "close": "19:00", "days": "mon-sat", "shift_hours": 8, "rotates": True},
    "nightlife": {"open": "18:00", "close": "02:00", "days": "all",   "shift_hours": 8, "rotates": True},
}

# ============================================================================
# 夜勤シフト(Wave 4 III-1「夜間開放」。**--night-shifts を渡したときだけ台帳に載る**)
# ----------------------------------------------------------------------------
# 監査の指摘: 上の 4 種は最遅でも close=23:00 で、**夜勤が語彙として存在しない**。
# ここは「渋谷の夜を実際に動かしている職種」を産業大分類へ割り付ける表で、
# close < open が「翌朝まで」= 日跨ぎシフトを意味する(読み手は society/work._window。
# world.night_economy.enabled が ON のときだけ日跨ぎとして解釈される)。
#
# 割付の根拠(実査): 渋谷区の夜間は ①ビルメンテナンス・警備(区内サービス業の最大職種群)
# ②事業系ごみ収集・深夜物流(深夜帯に集中)③深夜営業の飲食(バー)④24 時間営業の食品小売
# ⑤24 時間営業の娯楽・入浴施設 が主。いずれも既存の産業大分類に収まるので**新しい業種は
# 作らない**(台帳のスキーマを増やさない)。
#
# share = その社の従業者のうち夜勤に回る比率 / min_employees = これ未満の社には夜勤枠を作らない
# (1 人しか居ない事業所に 24 時間の交代要員を置かないための下限)。roles は夜勤枠だけに使う
# 役割名(日勤の role 巡回は 1 バイトも変えない)。
# ★決定論: 台帳側では乱数を 1 draw も引かない(全社に night_shift を載せ、実際に何人が夜勤に
#   なるかは従業者数 × share の丸めだけで決まる = 小さい社では自然に 0 人になる)。
NIGHT_SHIFT_BY_KEY: dict[str, dict] = {
    # サービス業(施設管理/メンテ・警備)= 建物の清掃・設備巡回・常駐警備
    "SV": {"open": "22:00", "close": "06:00", "days": "all", "shift_hours": 8,
           "rotates": True, "share": 0.30, "min_employees": 4,
           "roles": ["夜間清掃", "設備巡回", "常駐警備"]},
    # 運輸業・郵便業(配送/物流)= 事業系ごみ収集・深夜の積み込み
    "TR": {"open": "23:00", "close": "07:00", "days": "all", "shift_hours": 8,
           "rotates": True, "share": 0.25, "min_employees": 4,
           "roles": ["深夜配送", "回収", "夜間仕分け"]},
    # 宿泊業・飲食サービス業 = 深夜営業のバー・ダイニング
    "FB": {"open": "20:00", "close": "04:00", "days": "all", "shift_hours": 8,
           "rotates": True, "share": 0.18, "min_employees": 4,
           "roles": ["深夜ホール", "深夜キッチン"]},
    # 卸売業・小売業 = 24 時間営業の食品小売(深夜帯の店番)
    "WR": {"open": "22:00", "close": "06:00", "days": "all", "shift_hours": 8,
           "rotates": True, "share": 0.10, "min_employees": 5,
           "roles": ["深夜スタッフ"]},
    # 生活関連サービス業・娯楽業 = 24 時間営業の娯楽・入浴施設
    "LS": {"open": "22:00", "close": "06:00", "days": "all", "shift_hours": 8,
           "rotates": True, "share": 0.12, "min_employees": 5,
           "roles": ["夜間フロント", "夜間スタッフ"]},
}

# センサス較正専用の夜勤枠。★NIGHT_SHIFT_BY_KEY 本体には足さない(同表は「既定の産業大分類の
# 夜勤語彙」= INDUSTRY_SPEC のキーだけで閉じている、という不変条件をテストが固定しているため)。
# 娯楽業(--census でのみ生える AM)= ライブハウス・クラブ・カラオケの深夜営業。
# 日勤帯より夜勤帯が本業なので share は他業種より高い。
CENSUS_NIGHT_SHIFT_BY_KEY: dict[str, dict] = {
    "AM": {"open": "22:00", "close": "05:00", "days": "all", "shift_hours": 7,
           "rotates": True, "share": 0.40, "min_employees": 3,
           "roles": ["深夜フロア", "深夜バー", "夜間音響"]},
}


def night_shift_for(industry_key: str, employees: int) -> dict | None:
    """その社に載せる夜勤シフト(該当しなければ None)。**決定論・乱数ゼロ**。

    ``min_employees`` 未満の社には夜勤枠を作らない(1 人の店に 24 時間交代を置かない)。
    センサス較正でだけ生える業種(AM 娯楽業)は CENSUS_NIGHT_SHIFT_BY_KEY 側に置いてある
    (既定モードではその業種の社が 1 件も無いので、参照しても結果は変わらない)。"""
    spec = NIGHT_SHIFT_BY_KEY.get(str(industry_key)) or \
        CENSUS_NIGHT_SHIFT_BY_KEY.get(str(industry_key))
    if spec is None or int(employees) < int(spec["min_employees"]):
        return None
    return dict(spec)


def night_slot_count(company: dict, k: int) -> int:
    """その社の従業者スロット k のうち何枠を夜勤に回すか(決定論・乱数ゼロ)。

    台帳に ``night_shift`` が無い社は常に 0 = 既存の生成結果と 1 バイトも変わらない。
    丸めは四捨五入なので、share を下回る小さい社では自然に 0 枠になる。"""
    ns = company.get("night_shift")
    if not ns or k <= 0:
        return 0
    n = int(round(int(k) * float(ns.get("share", 0.0))))
    return max(0, min(int(k), n))


# 名称の手続き生成トークン(実在地名は可・実在企業名は使わない=R17)。
_PLACE_TOKENS = [
    "道玄坂", "宮益坂", "神南", "宇田川", "センター街", "公園通り", "桜丘", "円山", "松濤", "神泉",
    "南平台", "鶯谷", "渋谷", "ハチ公", "スクランブル", "明治通り", "井の頭通り", "キャットストリート",
    "奥渋", "富ヶ谷", "渋谷駅前", "文化村通り", "スペイン坂", "宮下", "並木橋", "道玄坂上", "神宮前", "松濤上",
]
_CONCEPT_TOKENS = [
    "ネクスト", "グロース", "アーバン", "ブルーム", "コネクト", "フロンティア", "クレスト", "ソレイユ",
    "ノヴァ", "ヴェルデ", "ミライ", "つばさ", "あおぞら", "ひまわり", "こだま", "さくら", "まほろば",
    "リンク", "ステラ", "オルカ", "ゼファー", "トパーズ", "フェリーチェ", "ルミナス",
]
_FORM_TOKENS = ["(株)", "(株)", "(株)", "(有)", "合同会社", "(同)"]

# 建物への束ね方(POI割付)。per_floor=1フロアあたりのテナント数の目安・prestige=大組織の優先度。
_BLD_PER_FLOOR = {
    "office": 3.0, "public": 2.0, "hotel": 1.0, "station": 2.0,
    "retail": 2.0, "generic": 1.6, "residential": 1.0,
}
_BLD_PRESTIGE = {
    "office": 6, "station": 5, "public": 5, "hotel": 4,
    "retail": 3, "generic": 2, "residential": 1, "house?": 0,
}


def _normalize(xs: list[float]) -> list[float]:
    s = float(sum(xs))
    return [x / s for x in xs] if s else xs


def _quota(total: int, shares: list[float]) -> list[int]:
    """total を shares(合計1)へ最大剰余法で割付(合計= total を厳守・決定論)。"""
    raw = [total * s for s in shares]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - base[i], -i), reverse=True)
    for k in range(rem):
        base[order[k % len(order)]] += 1
    return base


# ============================================================================
# センサス較正(Wave 4 IV-1。**--census を渡したときだけ効く**。無指定なら 1 バイトも変わらない)
# ----------------------------------------------------------------------------
# 入力 = scripts/calibrate_orgs_census.py が書く data/realworld/census_r3/station_area_industry.json。
# その `sim_buckets` は「令和3年経済センサス-活動調査 町丁目別表の駅周辺13町丁目実測」を
# 台帳の industry_key へ写像したもの(事業所数・従業者数)。ここで使うのは **2 本の実測列だけ**:
#   ① 産業別 事業所数 → 社数の割付(最大剰余法)
#   ② 産業別 従業者数 → 従業者質量の割付(最大剰余法)+ 産業内は既存の裾長サンプルの相対形を保存
# 規模帯(SIZE_BANDS)は町丁目表に無い次元なので、**帯は駆動側ではなく従業者数からの派生ラベル**
# になる(_band_of)。これが唯一の意味変更で、正直に meta と研究文書へ書いてある。
# ============================================================================
def _band_of(employees: int) -> str:
    """従業者数 → 規模帯ラベル(センサス較正では帯は駆動側でなく派生ラベル)。"""
    n = int(employees)
    for label, lo, hi in SIZE_BANDS:
        if lo <= n <= hi:
            return label
    return SIZE_BANDS[-1][0] if n >= SIZE_BANDS[-1][1] else SIZE_BANDS[0][0]


def load_census(path: Path) -> dict:
    """センサス集計 JSON を読む(`sim_buckets` 必須)。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not doc.get("sim_buckets"):
        raise ValueError(f"センサス集計に sim_buckets が無い: {path} "
                         "(scripts/calibrate_orgs_census.py で生成すること)")
    return doc


def census_specs(census: dict) -> list[dict]:
    """センサスに事業所が 1 件でもある産業だけを ALL_INDUSTRY_SPEC の並び順で返す。"""
    b = census["sim_buckets"]
    unknown = sorted(set(b) - {s["key"] for s in ALL_INDUSTRY_SPEC})
    if unknown:
        raise ValueError(f"台帳に定義の無い産業キーがセンサス集計にある: {unknown}")
    return [s for s in ALL_INDUSTRY_SPEC
            if int(b.get(s["key"], {}).get("establishments", 0) or 0) > 0]


def census_cat(it: dict, census_mode: bool) -> str:
    """職場POIカテゴリ。既定モードは INDUSTRY_SPEC の cat をそのまま(不変)。"""
    return CENSUS_CAT_OVERRIDE.get(it["key"], it["cat"]) if census_mode else it["cat"]


def _rescale_to_quota(raw: list[int], target: int) -> list[int]:
    """`raw` の相対形を保ったまま合計を `target` へ一致させる(最大剰余法・各社 >=1・乱数ゼロ)。

    裾長(1 人の店 〜 数百人の本社)の形は `raw` が持っているので、比を保って総和だけを
    センサスの従業者質量へ合わせる。`target` が件数未満なら全社 1 人(呼び出し側が meta に記録)。
    """
    n = len(raw)
    if n == 0:
        return []
    if target <= n:
        return [1] * n
    rest = target - n                              # 先に全社へ 1 人ずつ確保してから比例配分
    s = float(sum(raw)) or float(n)
    exact = [rest * (r / s) for r in raw]
    base = [int(x) for x in exact]
    rem = rest - sum(base)
    order = sorted(range(n), key=lambda i: (exact[i] - base[i], -i), reverse=True)
    for k in range(rem):
        base[order[k % n]] += 1
    return [1 + b for b in base]


def _apply_census_employees(companies: list[dict], spans: list[tuple[int, int]],
                            specs: list[dict], census: dict, count: int,
                            night_shifts: bool) -> dict:
    """産業ごとの従業者質量をセンサス実測比へ割り直す(決定論・乱数ゼロ)。戻り値=検収用の要約。"""
    b = census["sim_buckets"]
    tot_est = sum(int(b[s["key"]]["establishments"]) for s in specs)
    tot_emp = sum(int(b[s["key"]]["employees"]) for s in specs)
    target_total = int(round(count * tot_emp / tot_est)) if tot_est else 0
    emp_by_ind = _quota(target_total, _normalize([float(b[s["key"]]["employees"]) for s in specs]))
    degenerate = []
    for it, (start, end), e_i in zip(specs, spans, emp_by_ind):
        chunk = companies[start:end]
        if not chunk:
            continue
        if e_i <= len(chunk):
            degenerate.append(it["key"])
        raw = [int(c["size"]["employees"]) for c in chunk]
        for c, emp in zip(chunk, _rescale_to_quota(raw, e_i)):
            c["size"] = {"employees": emp, "band": _band_of(emp)}
            c.pop("night_shift", None)
            if night_shifts:
                ns = night_shift_for(c["industry_key"], emp)
                if ns is not None:
                    c["night_shift"] = ns
    return {"target_employees_total": target_total,
            "census_employees_per_establishment": round(tot_emp / tot_est, 3) if tot_est else 0.0,
            "employees_by_industry_target": {s["key"]: e for s, e in zip(specs, emp_by_ind)},
            "degenerate_industries": degenerate}


def _sample_employees(rng: random.Random, band: tuple[str, int, int]) -> int:
    """規模帯内の従業者数をサンプル。300+ は log 一様(平均~650)で長い裾を作る。"""
    _, lo, hi = band
    if lo >= 300:
        return int(round(300 * (4.0 ** rng.random())))  # [300,1200], 平均~649
    return rng.randint(lo, hi)


def _gen_company_name(rng: random.Random, it: dict,
                      real_names: set[str], seen: set[str]) -> str:
    """架空の社名を手続き生成。実在の建物/POI名(OSM由来)は避ける(R17)。"""
    words = it["words"]
    for attempt in range(64):
        form = _FORM_TOKENS[rng.randrange(len(_FORM_TOKENS))]
        place = _PLACE_TOKENS[rng.randrange(len(_PLACE_TOKENS))]
        word = words[rng.randrange(len(words))]
        mid = _CONCEPT_TOKENS[rng.randrange(len(_CONCEPT_TOKENS))] if rng.random() < 0.6 else ""
        name = f"{form}{place}{mid}{word}"
        if name not in seen and name not in real_names:
            seen.add(name)
            return name
    # 万一の衝突は決定論の数字サフィックスで一意化(実在名は依然回避)。
    n = 2
    while True:
        cand = f"{name}・{n}"
        if cand not in seen and cand not in real_names:
            seen.add(cand)
            return cand
        n += 1


def build_companies_dist(real_names: set[str], count: int, seed: int,
                         night_shifts: bool = False,
                         census: dict | None = None,
                         out_plan: dict | None = None) -> list[dict]:
    """産業大分類×規模帯の目標分布から count 社を決定論生成(workplace_poi は未割付)。

    night_shifts=True(CLI --night-shifts)のときだけ、該当業種の社に ``night_shift`` を
    1 キー足す(Wave 4 III-1)。**乱数は 1 draw も引かない**ので、既定 False の出力は
    従来と完全に同一(社名も規模も割付も 1 バイトも動かない)。

    census(CLI --census)を渡すと、産業別の社数と従業者質量が経済センサス実測比になる
    (Wave 4 IV-1)。**census=None の経路は上と同一のコードを通る**ので、無指定の出力は
    従来と 1 バイトも変わらない(tests/test_census_calibration.py がバイト一致を固定)。
    """
    rng = random.Random(seed)
    census_mode = census is not None
    specs = census_specs(census) if census_mode else INDUSTRY_SPEC
    if census_mode:
        shares = _normalize([float(census["sim_buckets"][it["key"]]["establishments"])
                             for it in specs])
    else:
        shares = _normalize([it["share"] for it in INDUSTRY_SPEC])
    n_by_ind = _quota(count, shares)
    companies: list[dict] = []
    seen_names: set[str] = set()
    spans: list[tuple[int, int]] = []      # 産業ごとの [開始, 終了) = 従業者再割付の単位
    seq = 0
    for it, n_i in zip(specs, n_by_ind):
        it = _census_spec(it, census_mode)          # 既定モードでは同一オブジェクト(不変)
        start = len(companies)
        band_dist = _normalize(SIZE_CLASS_DIST[it["cls"]])
        n_by_band = _quota(n_i, band_dist)
        for bidx, n_ib in enumerate(n_by_band):
            band = SIZE_BANDS[bidx]
            for _ in range(n_ib):
                seq += 1
                emp = _sample_employees(rng, band)
                name = _gen_company_name(rng, it, real_names, seen_names)
                detail = it["details"][rng.randrange(len(it["details"]))]
                # センサス較正では従業者数が後段で割り直されるので、夜勤枠もその後に判定する。
                night = night_shift_for(it["key"], emp) if (night_shifts and not census_mode) else None
                companies.append({
                    "id": f"co_{it['key'].lower()}_{seq:05d}",
                    "name": name,
                    "industry": it["name"],
                    "industry_key": it["key"],
                    "sector_detail": detail,
                    "products_services": list(it["products"]),
                    "size": {"employees": emp, "band": band[0]},
                    "wage_tier": it["tier"],
                    "roles": list(it["roles"]),
                    "generalist_roles": list(it["generalist"]),
                    "output_kinds": list(it["output"]),
                    "shift_pattern": dict(SHIFT_BY_CAT[it["cat"]]),
                })
                if night is not None:              # 夜勤枠(--night-shifts のときだけ生える)
                    companies[-1]["night_shift"] = night
        spans.append((start, len(companies)))
    if census_mode:
        plan = _apply_census_employees(companies, spans, specs, census, count, night_shifts)
        if out_plan is not None:                   # 検収用の要約(meta へ載せる。台帳本体は不変)
            out_plan.update(plan)
    return companies


def load_buildings(map_path: Path) -> tuple[list[dict], set[str]]:
    """地図から建物リストと、実在名(建物+POI名。R17回避用)の集合を返す。"""
    d = json.loads(map_path.read_text(encoding="utf-8"))
    buildings = list(d.get("buildings", []))
    real_names: set[str] = set()
    for b in buildings:
        if b.get("name"):
            real_names.add(str(b["name"]))
    for p in d.get("pois", []):
        if p.get("name"):
            real_names.add(str(p["name"]))
    return buildings, real_names


def _building_slots(buildings: list[dict], kinds: set[str]) -> list[dict]:
    """kinds に含まれる建物を prestige 降順に並べ、建物×階のテナント枠(slot)を展開。"""
    pool = [b for b in buildings if b.get("kind") in kinds]
    pool.sort(key=lambda b: (-_BLD_PRESTIGE.get(b.get("kind", ""), 0),
                             -int(b.get("levels", 1) or 1), str(b.get("id", ""))))
    slots: list[dict] = []
    for b in pool:
        kind = b.get("kind", "generic")
        levels = max(1, int(b.get("levels", 1) or 1))
        cap = max(1, round(_BLD_PER_FLOOR.get(kind, 1.0) * levels))
        node = str(b.get("entrance") or b.get("id"))
        x = round(float(b.get("cx", 0.0)), 1)
        y = round(float(b.get("cy", 0.0)), 1)
        for k in range(cap):
            slots.append({"building": str(b["id"]), "node": node,
                          "floor": 1 + (k % levels), "x": x, "y": y})
    return slots


def place_on_buildings(companies: list[dict], buildings: list[dict],
                       cat_by_key: dict[str, str] | None = None) -> int:
    """各社を建物×階へ決定論束縛(大組織→高prestige/高層ビル優先)。戻り値=割付済み数。

    商業系建物のテナント枠が不足する場合のみ、住宅系建物(house?)へ溢れ分を割り付ける。
    `cat_by_key`(--census のときだけ渡す)は職場POIカテゴリの差し替え表。None = 従来と同一。
    """
    commercial = set(_BLD_PER_FLOOR.keys())
    slots = _building_slots(buildings, commercial)
    if len(slots) < len(companies):
        slots += _building_slots(buildings, {"house?"})  # 溢れ分の予備(prestige最下位)
    order = sorted(range(len(companies)),
                   key=lambda j: (-int(companies[j]["size"]["employees"]), companies[j]["id"]))
    override = dict(cat_by_key or {})
    placed = 0
    for rank, j in enumerate(order):
        if rank >= len(slots):
            break
        sl = slots[rank]
        key = companies[j]["industry_key"]
        cat = override.get(key) or IND_BY_KEY[key]["cat"]
        companies[j]["workplace_poi"] = {
            "cat": cat, "building": sl["building"], "node": sl["node"],
            "floor": sl["floor"], "x": sl["x"], "y": sl["y"],
        }
        placed += 1
    return placed


def _dist_stats(companies: list[dict]) -> dict:
    """生成台帳の実現分布(産業別件数・従業者・規模帯)を集計(meta記録+検収用)。"""
    n = len(companies)
    total_emp = sum(int(c["size"]["employees"]) for c in companies)
    by_ind: dict[str, dict] = {}
    by_band: dict[str, int] = {b[0]: 0 for b in SIZE_BANDS}
    for c in companies:
        key = c["industry_key"]
        d = by_ind.setdefault(key, {"industry": c["industry"], "count": 0, "employees": 0})
        d["count"] += 1
        d["employees"] += int(c["size"]["employees"])
        by_band[c["size"]["band"]] += 1
    for key, d in by_ind.items():
        d["count_share"] = round(d["count"] / n, 4) if n else 0.0
        d["employee_share"] = round(d["employees"] / total_emp, 4) if total_emp else 0.0
    return {
        "n_companies": n, "employees_total": total_emp,
        "employees_per_org": round(total_emp / n, 2) if n else 0.0,
        "by_industry": by_ind,
        "size_band_counts": by_band,
        "size_band_shares": {k: (round(v / n, 4) if n else 0.0) for k, v in by_band.items()},
    }


def build_ledger_dist(map_path: Path, count: int, seed: int,
                      night_shifts: bool = False, census: dict | None = None,
                      census_file: str = "") -> dict:
    """分布駆動の組織台帳(会社~count + 学校)を組み立てて返す。

    census(--census)を渡すと産業構成が経済センサス実測比になる。None = 従来と完全同一。
    """
    buildings, real_names = load_buildings(map_path)
    plan: dict = {}
    companies = build_companies_dist(real_names, count, seed, night_shifts, census, plan)
    census_mode = census is not None
    specs = census_specs(census) if census_mode else INDUSTRY_SPEC
    cat_by_key = {it["key"]: census_cat(it, True) for it in specs} if census_mode else None
    placed = place_on_buildings(companies, buildings, cat_by_key)
    # 学校ブロックは既存の代表校キュレーション(build_orgs)をそのまま再利用(改変なし)。
    by_cat, _ = load_map(map_path)
    schools = build_orgs(by_cat, seed)["schools"]
    stats = _dist_stats(companies)
    if census_mode:
        shares = _normalize([float(census["sim_buckets"][it["key"]]["establishments"])
                             for it in specs])
    else:
        shares = _normalize([i["share"] for i in INDUSTRY_SPEC])
    target_ind = {it["key"]: {"industry": _census_spec(it, census_mode)["name"],
                              "target_count_share": round(s, 4), "size_class": it["cls"]}
                  for it, s in zip(specs, shares)}
    doc = {
        "meta": {
            "generator": "scripts/build_orgs.py --dist" + (" --census" if census_mode else ""),
            "mode": "census-calibrated" if census_mode else "distribution-driven",
            "seed": seed, "target_count": count,
            "map": str(map_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "note": ("架空の組織台帳(R17: 実在企業名・学校名なし)。産業大分類×従業者規模帯の"
                     "分布から seed 付き決定論で手続き生成。名称は業種語彙×実在地名×概念語の合成で、"
                     "OSM由来の実在名は回避。組織は建物×階へ束縛(オフィスビル=多数・路面店=少数)。"),
            "honesty": ("事業所数構成比=東商渋谷支部(平18)骨格 / 規模帯=全国経済センサスの裾長構造を"
                        "渋谷bbox実測(事業所~1.19万・従業者~25.7万→平均~21.6)へ補正した近似。"
                        "産業×規模のクロス実表(e-Stat stat_infid=000031720779)は静的抽出不可のため"
                        "再現を主張しない。産業別従業者シェアは生成の副産物(近似)。"),
            "sources_doc": "docs/research/shibuya-population.md, docs/research/shibuya-organizations.md",
            "anchors": {"target_establishments": count, "target_employees": 257000,
                        "employees_tolerance": "±10%"},
            "work_hours_standard": WORK_HOURS,
            # Wave 4 III-1: 夜勤枠を載せたか(false の台帳からは夜勤者が 1 人も生まれない)。
            "night_shifts": bool(night_shifts),
            "target_industry_distribution": target_ind,
            "realized": stats,
            "poi_coverage": round(placed / max(1, len(companies)), 4),
            "counts": {"companies": len(companies), "schools": len(schools)},
        },
        "companies": companies,
        "schools": schools,
    }
    if census_mode:
        # ★センサス較正のときだけ足すブロック(既定モードの meta は 1 キーも増えない)。
        cm = census.get("_meta", {})
        area = census.get("area", {})
        cat_counts: dict[str, int] = {}
        for c in companies:
            cat_counts[cat_by_key[c["industry_key"]]] = cat_counts.get(
                cat_by_key[c["industry_key"]], 0) + 1
        doc["meta"]["census"] = {
            "file": census_file,
            "source": cm.get("source", ""),
            "attribution": cm.get("attribution", ""),
            "survey": cm.get("survey", {}),
            "area": {"name": (cm.get("area_definition") or {}).get("name", ""),
                     "blocks": list(area.get("blocks", [])),
                     "establishments": area.get("establishments"),
                     "employees": area.get("employees")},
            "employee_plan": plan,
            "workplace_cat_counts": cat_counts,
            "cat_overrides": {k: v for k, v in CENSUS_CAT_OVERRIDE.items()
                              if k in {s["key"] for s in specs}},
            "census_only_industries": [s["key"] for s in CENSUS_EXTRA_INDUSTRY_SPEC
                                       if s["key"] in {t["key"] for t in specs}],
        }
        doc["meta"]["honesty"] = (
            "産業別の事業所数・従業者数は令和3年経済センサス-活動調査 町丁目別表の"
            "駅周辺13町丁目実測(scripts/calibrate_orgs_census.py)。社数は事業所数シェア、"
            "産業別の従業者総和は従業者シェアの最大剰余法割付で実測比に一致する。"
            "★規模帯(SIZE_BANDS)は町丁目表に無い次元なので、帯は駆動側ではなく従業者数からの"
            "派生ラベルであり、産業内の規模分布(裾の形)は依然として全国構造からの近似である"
            "(社ごとの従業者数の実表を再現したとは主張しない)。")
        doc["meta"]["anchors"] = {
            "target_establishments": count,
            "target_employees": plan.get("target_employees_total"),
            "census_establishments": area.get("establishments"),
            "census_employees": area.get("employees"),
            "employees_tolerance": "0(最大剰余法で完全一致)"}
        doc["meta"]["sources_doc"] = "docs/research/economy-census-calibration.md"
    return doc


def write_generation_report(ledger: dict, out_path: Path) -> None:
    """生成レポート(docs)。正直な近似の根拠+実現分布の数値を残す。"""
    m = ledger["meta"]; st = m["realized"]
    lines = ["# 組織台帳 ~1.1万 生成レポート(P6・分布駆動)", "",
             f"- 生成器: `{m['generator']}` / seed={m['seed']} / map=`{m['map']}`",
             f"- 会社 {st['n_companies']} 社 + 学校 {m['counts']['schools']} 校",
             f"- 従業者総和: **{st['employees_total']:,}** "
             f"(目標~{int(m.get('anchors', {}).get('target_employees') or 0):,}・"
             f"平均 {st['employees_per_org']} 人/社)",
             f"- POI(建物)割付率: **{m['poi_coverage']:.1%}**", "",
             "## 産業大分類の分布(目標事業所数シェア → 実現)", "",
             "| key | 産業 | 規模クラス | 目標件数シェア | 実現件数 | 実現件数シェア | 従業者 | 従業者シェア |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, tgt in m["target_industry_distribution"].items():
        r = st["by_industry"].get(key, {"count": 0, "count_share": 0, "employees": 0, "employee_share": 0})
        lines.append(f"| {key} | {tgt['industry']} | {tgt['size_class']} | "
                     f"{tgt['target_count_share']:.3f} | {r['count']} | {r['count_share']:.3f} | "
                     f"{r['employees']:,} | {r['employee_share']:.3f} |")
    lines += ["", "## 従業者規模帯の分布(実現)", "",
              "| 規模帯 | 件数 | シェア |", "|---|---:|---:|"]
    for b in SIZE_BANDS:
        lines.append(f"| {b[0]} | {st['size_band_counts'][b[0]]} | {st['size_band_shares'][b[0]]:.3f} |")
    lines += ["", "## 正直な近似の記録", "", m["honesty"], ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _run_dist_mode(args) -> None:
    """分布駆動モード(P6): 経済センサス分布から ~1.1万組織を生成し新ファイルへ書き出す。"""
    map_path = Path(args.map) if args.map else DEFAULT_WIDE_MAP
    if not map_path.is_absolute():
        map_path = REPO_ROOT / map_path
    census = census_file = None
    if getattr(args, "census", None):
        cpath = Path(args.census)
        if not cpath.is_absolute():
            cpath = REPO_ROOT / cpath
        census = load_census(cpath)
        try:
            census_file = str(cpath.relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:                              # リポジトリ外の指定もそのまま記録する
            census_file = str(cpath)
    ledger = build_ledger_dist(map_path, int(args.count), int(args.seed),
                               bool(getattr(args, "night_shifts", False)),
                               census, census_file or "")
    # --census は既定の出力先を分けて、正準 11k 台帳を事故で上書きしない。
    default_out = DEFAULT_CENSUS_OUT if census is not None else DEFAULT_DIST_OUT
    out = Path(args.orgs_out) if args.orgs_out else default_out
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
    st = ledger["meta"]["realized"]
    print(f"written: {out}")
    print(f"  会社 {st['n_companies']} / 学校 {ledger['meta']['counts']['schools']}  "
          f"従業者総和 {st['employees_total']:,} (平均 {st['employees_per_org']}/社)  "
          f"POI割付 {ledger['meta']['poi_coverage']:.1%}")
    if args.report:
        default_rep = (REPO_ROOT / "docs" / "research" / "org-book-census.md"
                       if census is not None else DEFAULT_REPORT)
        rep = Path(args.report) if isinstance(args.report, str) else default_rep
        if not rep.is_absolute():
            rep = REPO_ROOT / rep
        write_generation_report(ledger, rep)
        print(f"written: {rep}")


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
    # --- 分布駆動モード(P6。既存の呼び出し方は不変) ---
    ap.add_argument("--dist", action="store_true",
                    help="分布駆動モード: 経済センサス分布から ~1.1万組織を生成し"
                         " data/organizations_shibuya_wide11k.json へ出力(既存台帳は上書きしない)")
    ap.add_argument("--count", type=int, default=11000, help="--dist で生成する組織数(既定 11000)")
    ap.add_argument("--report", action="store_true",
                    help="--dist の生成レポートを docs/research/org-book-11k.md へ出力")
    ap.add_argument("--night-shifts", action="store_true",
                    help="夜勤シフト(22:00-06:00 の施設管理/警備・23:00-07:00 の深夜物流・"
                         "20:00-04:00 の深夜飲食 等)を台帳に載せる(Wave 4 III-1)。"
                         "**既定 OFF = 従来と 1 バイトも変わらない台帳**。ON にした台帳から"
                         "プールを再生成すると L2 に夜勤者が生まれる")
    ap.add_argument("--census", nargs="?", const=str(DEFAULT_CENSUS_JSON), default=None,
                    help="経済センサス(令和3)駅周辺集計 JSON で産業構成を較正する(Wave 4 IV-1)。"
                         "既定パス data/realworld/census_r3/station_area_industry.json は"
                         "scripts/calibrate_orgs_census.py が生成する。**無指定 = 従来と 1 バイトも"
                         "変わらない台帳**。指定時は --dist を暗黙に立て、出力先も"
                         "data/organizations_shibuya_census.json へ分ける(正準台帳を上書きしない)")
    args = ap.parse_args()

    if args.dist or args.census:
        # 分布駆動モードは既定 --map を広域地図へ切替(明示指定があればそれを優先)。
        if args.map == str(DEFAULT_MAP):
            args.map = None
        _run_dist_mode(args)
        return

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
