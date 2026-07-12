# 渋谷シミュ × Unreal Engine 5 + PLATEAU — 完全手順書

シミュ結果(エージェント・車の動き)を、PLATEAU の**実形状の渋谷の街**に載せて
フォトリアルに再生し、視点を自由に動かすための、UE 初めての人向けゼロからの手順。

- 可視化の**メイン経路**= Unreal Engine 5 + PLATEAU SDK for Unreal(本書)。
- Blender / Web3D(`viz/blender_import.py`, `viz/make_viewer3d.py`)は quick-look として維持。
- 座標変換・軌跡整形は `scripts/export_ue.py` に一本化(UE 側は素直な再生器)。

> ⚠️ この手順は UE 実機で未検証(開発環境に UE 無し)。手順は一次情報に基づくが、
> UI 文言・API はバージョン差があり得る。詰まったら §8 トラブルシュートと各ソース URL を参照。

---

## 0. 必要環境と対応バージョン(2026-07 時点の一次情報)

| 要素 | 推奨 | 出典 |
|---|---|---|
| Unreal Engine | **5.5.4**(PLATEAU SDK for Unreal v3.2.x が対応) | SDK Release ページ |
| PLATEAU SDK for Unreal | **v3.2.2**(2025-06-03)。CityGML v3 まで対応、渋谷 2025 の CityGML v5 も読込可 | GitHub Releases |
| 渋谷データ | **3D都市モデル 渋谷区(2025年度)** CityGML / 3D Tiles | G空間情報センター |
| GPU | Lumen/Nanite を使うので RTX 2060 相当以上を推奨 | — |

- v3.1.1 以前は UE 5.3.2 対応。**5.5 系を使うなら SDK は v3.2.0 以降**。
- インストール: Fab(旧マーケットプレイス)版、または GitHub Releases の `.zip` をプロジェクトの
  `Plugins/` に置く。エディタ再起動で `PLATEAU` メニューが出れば OK。

---

## 1. 渋谷データの入手(G空間情報センター)

データセット: **3D都市モデル(Project PLATEAU)渋谷区(2025年度)**
`https://www.geospatial.jp/ckan/dataset/plateau-13113-shibuya-ku-2025`

ダウンロードするファイル(いずれも PLATEAU Site Policy、商用可・無償):
- `13113_shibuya-ku_pref_2025_citygml_1_op.zip` … **CityGML(SDK でこれを読む)** LOD0/1/2.0/2.2
- `13113_shibuya-ku_pref_2025_3dtiles_mvt_1_op.zip` … 3D Tiles/MVT(Web/Cesium 用。UE では使わない)
- `13113_shibuya-ku_2025_related.zip` … 付属(コードリスト等)

- **FBX は配布されない**方針。UE には CityGML を SDK で直接読ませる(変換不要)。
- 展開すると `udx/`(地物ごとフォルダ: `bldg`=建物, `tran`=道路 …)と `codelists/` がある。
  SDK では **`udx` の 1 つ上の階層フォルダ**を指定する。

---

## 2. UE プロジェクト作成 → PLATEAU 都市の読込み

1. UE 5.5 で **Games > Blank**(または Third Person)テンプレートの新規プロジェクト。
2. PLATEAU SDK for Unreal プラグインを有効化(§0)。エディタ再起動。
3. メニュー **PLATEAU → PLATEAU SDK** → **インポート**タブ。
4. インポート元 = **ローカル**、`udx` の 1 つ上のフォルダを選択。
5. **基準座標系の選択**: 渋谷は関東 → **第9系(EPSG:6677 / JGD2011 平面直角第IX系)** を選ぶ。
6. **基準座標系からのオフセット値の設定**(重要・§4 で使う):
   東西・南北・高さ [m] を入力。ここを **スクランブル交差点の平面直角座標**にすると、
   PLATEAU 原点(レベルの 0,0,0)がスクランブル交差点に一致する。値は §4。
7. **最小/最大 LOD**: 建物は **LOD2(最大 2.2)**、道路 `tran` は LOD1 で十分。
8. **モデル結合単位**: 主要地物単位(建物ごとに選択できて扱いやすい)。
9. インポート実行 → CityGML がメッシュ化されてレベルに配置される(数分)。
   - CityGML は直接描画できないため、SDK が取り込み時にポリゴンメッシュへ変換する。

---

## 3. シミュ側の書き出し(この環境で実行)

UE に渡す軌跡を作る。**UE の外(この Python 環境)で 1 コマンド**:

```bash
# 1) 中立 3D シーンを作る(未生成なら)
python scripts/export_3d.py runs/<name>

# 2) UE 用に座標変換して sim_ue.json を出す
python scripts/export_ue.py runs/<name>
#   → runs/<name>/scene3d/sim_ue.json(既に UE cm 座標)
#   位置合わせを変えたい時だけ引数を足す(§4/§8):
#   python scripts/export_ue.py runs/<name> --heading 90 --offset 0 0 0 --csv
```

`sim_ue.json` は **すでに UE ワールド座標(cm, Z-up, 左手系)**。UE 側は変換しない。

---

## 4. 座標合わせ(sim ↔ PLATEAU)— ここが要

### 4.1 3 つの座標系
- **sim(local-m)**: X=east, Y=north, Z=up [m]、**右手系**、原点=スクランブル交差点
  `(lat 35.65950, lon 139.70062)`。
- **UE**: X=前, Y=右, Z=上 [cm=uu]、**左手系**、1uu=1cm。
- **PLATEAU(平面直角 第9系)**: X=北距, Y=東距 [m]。SDK が UE cm へ載せる。

### 4.2 いちばん簡単な合わせ方(推奨)
**PLATEAU 原点 = スクランブル交差点**にしてしまえば、sim 原点(0,0,0)とそのまま重なる。
そのために §2-6 の SDK オフセットへ、スクランブル交差点の平面直角座標(EPSG:6677 第9系)を入れる:

```
東西(Easting  / Y) = -12015.952  m
南北(Northing / X) =  -37768.576 m
高さ               =  0           m(地面基準。PLATEAU の標高は §4.4 で吸収)
```

(この値は `scripts/export_ue.py` の `ORIGIN_EPSG6677` にも記録。Kawase 2011 の
Transverse Mercator 級数で算出、GSI 準拠。)

こうすると `export_ue.py` は **`--offset 0 0 0`(既定)のまま**で sim 原点が PLATEAU の
スクランブル交差点に載る。

### 4.3 向き(heading)と鏡像(y_flip)は実測合わせ
- SDK が平面直角(X=北, Y=東)を UE の X/Y にどう割り当てるかはバージョン差があるため、
  **北の向き**は初回にエディタで実測して合わせる。手順:
  1. `export_ue.py`(既定 heading=0, y_flip=True)で出力し、§5 で ISM を配置。
  2. **ハチ公像(交差点南西)や渋谷駅(南東)** など非対称ランドマークと、sim の建物配置を見比べる。
  3. ズレていたら `--heading 90 / 180 / 270` を試し、**鏡像**(左右反転)なら `--no-yflip` を付けて再出力。
     4 通り×反転有無=8 通りのどれかで必ず合う(アフィン変換なので）。
- 合ったら、その `--heading`/`--yflip` を確定値として運用に固定する。

### 4.4 高さ(標高)
- PLATEAU 建物は **標高(T.P.)基準**、sim は地面 z=0 基準。渋谷は起伏があるので、
  スクランブル交差点付近の標高ぶん、PLATEAU 側 Z をオフセットするか、
  `--offset 0 0 <Z_uu>` でエージェントを持ち上げて路面に合わせる(交差点標高 ≈ 数十 m ×100uu)。
  厳密でなくても、人が地面から浮く/埋まる量を見て 1 回調整すれば足りる。

---

## 5. UE へ取り込み(エディタ内 Python)

`viz/unreal/import_shibuya_sim.py` を UE の Python から呼ぶ。

1. UE メニュー **Tools → Execute Python Script**、または **Output Log の Cmd を Python に切替**。
2. Python コンソールで:
   ```python
   import sys; sys.path.append(r"C:/…/shibuya-simulation/viz/unreal")
   import import_shibuya_sim as imp
   imp.run(r"C:/…/shibuya-simulation/runs/<name>/scene3d")          # ISM 方式(推奨)
   ```
   - `mode="ism"`(既定): **人 ISM + 車 ISM** を持つ `ShibuyaSimReplay` アクタを配置し、
     `sim_ue.json` を `Content/ShibuyaSim/` にコピー。→ §6 の SimReplayActor が再生。
   - `mode="sequence"`: 小規模(<=~300体)。1 体=1 アクタ + Level Sequence にキーをベイク(映像用)。
     ```python
     imp.run(r"C:/…/runs/<name>/scene3d", mode="sequence", max_seq_actors=300)
     ```
- Python プラグイン(**Python Editor Script Plugin**)を有効化しておくこと。

---

## 6. ランタイム再生(SimReplayActor)

設計と擬似コードは **`viz/unreal/SimReplayActor_DESIGN.md`**。要点:
- `Content/ShibuyaSim/sim_ue.json` を BeginPlay で 1 度だけパース。
- 毎 tick、`posAt` 相当の道なり補間で **人 ISM / 車 ISM を一括更新**(`BatchUpdateInstancesTransforms`)。
- `sim_min` から太陽(SunSky)を回して**昼夜連動**(ビューアと同じ角度モデル)。
- C++ が本命だが、**数百〜千体なら Blueprint のみ**でも実装可(標準 `JsonBlueprintUtilities` で JSON 読込)。

`import_shibuya_sim.py mode="ism"` はアクタと ISM・JSON を用意するところまで。
再生ロジック(BP/C++)は上記設計に沿って 1 度作れば、以後は JSON 差し替えで別ランを再生できる。

---

## 7. 昼夜ライティングとレンダ設定

### 7.1 昼夜(sim 時刻 → 太陽)
- SDK/エンジンの **SunSky** アクタを置き、`SimReplayActor` が `sim_min` から Solar Time を毎tick更新。
- 角度・昼度は `viz/make_viewer3d.py updateSky()` と同一式(6:00 日の出付近、正午に仰角最大)。
  → Web/Blender/UE で時刻ごとの見えが揃う。

### 7.2 推奨レンダ(Lumen)
- **Global Illumination = Lumen**、**Reflections = Lumen**(Project Settings)。
- **Nanite**: PLATEAU LOD2 建物メッシュに有効化すると大量ポリゴンでも軽い。
- **SkyAtmosphere + Volumetric Cloud + Exponential Height Fog** で渋谷の空気感。
- ポストプロセスで Exposure(Auto)・Bloom 控えめ。夜は建物の窓 Emissive で"点灯"。
- 映像書き出しは **Movie Render Queue**(高品質・Path Tracer も可)。

---

## 8. トラブルシュート

| 症状 | 対処 |
|---|---|
| 街と人がズレる/鏡像 | §4.3。`--heading {0,90,180,270}` × `--no-yflip` の 8 通りを試す |
| 人が地面から浮く/埋まる | §4.4。`--offset 0 0 <Z_uu>` で高さ調整。シリンダ原点補正 `+AGENT_H/2` の符号も確認 |
| 人が超巨大/極小 | 単位ズレ。`export_ue.py` の `--scale`(既定100=m→cm)を確認。PLATEAU も cm 前提 |
| `add_component_by_class` が無い | UE バージョン差。BP で ISM を 2 つ持つ親アクタを用意し、Python はインスタンス追加のみに |
| PLATEAU メニューが出ない | プラグイン未有効/UE-SDK バージョン不一致(§0)。UE5.5↔SDK v3.2.x を合わせる |
| インポートで udx が選べない | `udx` の **1 つ上**のフォルダを選ぶ(`udx/` と `codelists/` が見える階層) |
| 数千体でカクつく | ISM→**Niagara + VAT** へ(§9)。BP 走査なら C++ 化 |

---

## 9. 大量エージェント(80〜1万体)の手法

体数レンジで手法を切り替える(`SimReplayActor_DESIGN.md` の ISM が基本、その先):

- **〜数百体**: 1 体=StaticMeshActor + Level Sequence でも可(`mode="sequence"`)。映像向き。
- **数百〜数千体**: **ISM/HISM + ランタイム更新**(本実装の推奨)。1 メッシュ 1 ドローコール。
  歩行アニメが要るなら **AnimToTexture(VAT: Vertex Animation Texture)** を焼いて ISM に適用
  (アニメを GPU 頂点テクスチャ化、CPU 負荷を抑えて大量再生)。
- **数千〜1万体**: **Niagara + VAT**(例: OverCrowd 系)。位置は Niagara の Data Channel/配列へ渡し、
  VAT で歩行を表現。最大百万体級の実例あり。
- 参考: UE 標準の **Mass(MassEntity/StateTree)** はエージェント AI/群集の枠組み。ただし本件は
  「外部シミュの軌跡再生」なので **位置は sim_ue.json が唯一の真実**。Mass の挙動生成は使わず、
  描画スケール手段(ISM/VAT/Niagara)だけ借りるのが素直。

---

## 10. データの日付整合(後続の地図バッチへ)

PLATEAU 渋谷 2025 の**実測基準日(航空写真の撮影年月)**はカタログ頁に明記が無い
(カタログ上の作成日 2026-03-16 / 更新日 2026-04-03、年度=2025年度)。街の実形状に
シミュの地図(OSM 由来)を整合させるには、**OSM 取得日を PLATEAU 基準日に寄せる**:

- Overpass の attic(過去日)クエリ: ヘッダに `[date:"YYYY-MM-DDThh:mm:ssZ"]` を付けるとその
  時点の DB 状態を返す。例(渋谷 bbox・建物):
  ```
  [out:json][timeout:60][date:"2025-04-01T00:00:00Z"];
  (way["building"](35.653,139.694,35.665,139.706);
   relation["building"](35.653,139.694,35.665,139.706););
  out body geom;
  ```
- 詳細な結論日と根拠は `docs/research/3d-visualization.md` §6 を参照(この文書と一対)。

---

## ソース(一次情報)
- 渋谷2025 データ: <https://www.geospatial.jp/ckan/dataset/plateau-13113-shibuya-ku-2025>
- PLATEAU SDK for Unreal(GitHub / Releases): <https://github.com/Synesthesias/PLATEAU-SDK-for-Unreal> ・ <https://github.com/Project-PLATEAU/PLATEAU-SDK-for-Unreal/releases>
- インポート手順(公式マニュアル): <https://project-plateau.github.io/PLATEAU-SDK-for-Unreal/manual/ImportCityModels.html>
- TOPIC17 SDK活用(Unreal): <https://www.mlit.go.jp/plateau/learning/tpc17-2/>
- 単位(m→cm ×100): <https://www.mlit.go.jp/plateau/learning/tpc10-1/>
- Overpass 日付指定(attic): <https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL> ・ <https://wiki.openstreetmap.org/wiki/Attic_Data>
- 大量群集(UE): <https://vrealmatic.com/unreal-engine/crowds> ・ <https://jettelly.com/blog/simulate-massive-crowds-in-unreal-engine-5-with-overcrowd>
