# SimReplayActor 設計 — ランタイム再生方式(推奨)

渋谷シミュの軌跡(`sim_ue.json`)を UE5 ランタイムで再生する自作 Actor の設計。
`viz/unreal/import_shibuya_sim.py` の `mode="ism"` が用意する **2 つの InstancedStaticMeshComponent
(人・車)** を、毎 tick 位置更新して滑らかに動かす。座標変換・`w` デコードは
`scripts/export_ue.py` 側で済んでいるので、ここは **「配列を読んで線形補間して ISM を更新する」だけ**。

> この環境に UE が無いため実装は未検証。以下は C++/Blueprint で組める粒度の設計 + 擬似コード。

---

## 0. なぜ Sequencer ではなくランタイム Actor か(方式比較)

| 観点 | Level Sequence(Sequencer) | SimReplayActor(ランタイム ISM) ← 推奨 |
|---|---|---|
| 体数スケール | △ 1体=1アクタ。数百体で破綻(キー数・アクタ数) | ◎ ISM 1 ドローコール/メッシュ。80〜1万体を実測レンジ |
| ISM 個別インスタンス | ✕ Sequencer は ISM インスタンスを個別キーできない | ◎ `UpdateInstanceTransform` で毎tick更新が本領 |
| 補間の質 | キー間 UE 補間(打ち方次第) | ◎ 元 `posAt`(道なり `alongPath`)を忠実移植 |
| 昼夜・時刻連動 | 別途 | ◎ 同 Actor 内で `sim_min` から SunSky を駆動 |
| 映像書き出し(MRQ) | ◎ Sequencer が最短 | ○ Sequencer に「再生する空シーケンス」を噛ませれば MRQ 収録可 |
| セットアップ簡便 | ○ | ○(BP 1 個 + JSON パス) |

**結論**: インタラクティブ観察・大規模はランタイム ISM が本命。映像レンダは
`import_shibuya_sim.py mode="sequence"`(小規模)か、ランタイム Actor を再生しながら MRQ で収録。

---

## 1. データ契約(`sim_ue.json`、既に UE 座標)

```
meta: { units:"cm", up_axis:"Z", nSteps, step_minutes, start_min,
        floor_height_uu, sim_min:[...], car_ground_uu, scale_m_to_uu, offset_uu, heading_deg, y_flip }
agents: [ {id,name,visitor,occupation,age,gender,has_car,has_bicycle}, ... ]   # 色分け・UI 用
ids: [agent_id, ...]
positions: [step][i] = [ux, uy, uz, state]     # state: 0=路上 -1=範囲外 -2=睡眠 >=1000=屋内(uz は階高反映済み)
moves:     [step][i] = [mode, [[ux,uy],...]] | null   # mode: 0=walk 1=bicycle 2=car。歩行の道なり補間用
traffic:   [step]    = [ [[ux,uy],[ux,uy]], ... ]     # 車の線分(1 本=1 台ぶんの当該 step 経路)
```

`positions[s][i]` と `moves[s+1][i]` の関係、`state` の意味は 2D/3D ビューアと完全に一致
(`viz/make_viewer3d.py` の `posAt`/`upOf`)。C++ 側はこの意味論だけ知っていればよい。

---

## 2. クラス構成

```
AShibuyaSimReplay : public AActor
  UPROPERTY() FString JsonPath;                 // Content/ShibuyaSim/sim_ue.json(import スクリプトが配置)
  UPROPERTY() UInstancedStaticMeshComponent* AgentISM;   // 人(シリンダ/カプセル)
  UPROPERTY() UInstancedStaticMeshComponent* CarISM;     // 車(ボックス)
  UPROPERTY(EditAnywhere) float PlaybackSpeed = 1.2f;    // 1x ≈ 1.2 step/秒(ビューアと同じ体感)
  UPROPERTY(EditAnywhere) bool  bLoop = true;
  UPROPERTY(EditAnywhere) UDirectionalLightComponent* Sun;  // 任意。sim_min で昼夜
  // 解析済みデータ(TArray に展開。JSON は BeginPlay で一度だけパース)
  FSimData Data;   // positions/moves/traffic/sim_min/agents をネイティブ配列化
  double T = 0.0;  // 連続 step 位置(小数)
```

`FSimData` へ `BeginPlay` で JSON を展開しておく(毎 tick の JSON アクセスは厳禁)。
JSON パースは `FJsonSerializer`(エンジン標準)。数万要素でも起動時一括なら問題なし。

---

## 3. 主要ロジック(擬似コード)

### 3.1 BeginPlay — ロード & インスタンス確保
```
BeginPlay():
  FString raw; FFileHelper::LoadFileToString(raw, *JsonPath)
  Data = ParseSimUe(raw)                      // positions/moves/traffic/agents/meta
  AgentISM->SetStaticMesh(CylinderMesh)
  AgentISM->NumCustomDataFloats = 1           // 0=居住 1=来訪(マテリアルで色分け)
  for i in 0..NA-1:
     idx = AgentISM->AddInstance(HiddenXform())     // 初期は隠し
     AgentISM->SetCustomDataValue(idx, 0, Data.agents[i].visitor ? 1 : 0)
  CarISM->SetStaticMesh(CubeMesh)
  for c in 0..CAR_CAP-1: CarISM->AddInstance(HiddenXform())
  T = 0
```

### 3.2 Tick — 補間して ISM を一括更新
```
Tick(dt):
  if playing: T += dt * PlaybackSpeed
  if T >= nSteps-1: T = bLoop ? 0 : nSteps-1
  s0 = floor(T); f = T - s0

  # --- 人 ---
  TArray<FTransform> xf; xf.SetNum(NA)
  for i in 0..NA-1:
     (ux,uy,uz,state) = AgentPos(i, s0, f)          # ← 3.3
     if state == OFFAREA or state == SLEEP:
        xf[i] = HiddenXform()                        # 地下 or scale 0 で隠す
     else:
        FVector loc(ux, uy, uz + AGENT_H/2)          # シリンダは中心原点 → 足元合わせ
        FVector scl(AGENT_D/100, AGENT_D/100, AGENT_H/100)
        # 進行方向へ Yaw(任意): dir = normalize(next - cur); yaw = atan2(dir.Y, dir.X)
        xf[i] = FTransform(FRotator(0,yaw,0), loc, scl)
  AgentISM->BatchUpdateInstancesTransforms(0, xf, false, true, false)   # 1 コールで全更新

  # --- 車 ---
  segs = Data.traffic[s0]
  for c in 0..CAR_CAP-1:
     if c < segs.Num():
        p = AlongPath(segs[c], f)                     # 線分上を f で進む
        CarISM->UpdateInstanceTransform(c,
            FTransform(FRotator(0,carYaw,0), FVector(p.X,p.Y,car_ground_uu), CarScale), false, false)
     else:
        CarISM->UpdateInstanceTransform(c, HiddenXform(), false, false)
  CarISM->MarkRenderStateDirty()

  # --- 昼夜(任意) ---
  if Sun: UpdateSun(SimMinAt(T))                       # 3.4
```

### 3.3 AgentPos — `posAt` の C++ 版(道なり補間)
```
AgentPos(i, s0, f) -> (x,y,z,state):
  (ux,uy,uz,w) = Data.positions[s0][i]
  if w != STREET: return (ux,uy,uz,w)                  # 屋内/範囲外/睡眠は動かさない
  nm = (s0+1<NS) ? Data.moves[s0+1][i] : null
  if nm.valid:
     (px,py) = AlongPath(nm.pts, f); return (px,py,uz,STREET)   # 歩行=ポリライン道なり
  b = Data.positions[min(s0+1,NS-1)][i]
  if b.state != STREET: return (ux,uy,uz,STREET)
  return (lerp(ux,b.x,f), lerp(uy,b.y,f), uz, STREET)  # 直線補間フォールバック

AlongPath(pts, f):   # 総長 f の位置。export_ue._along / viewer3d.alongPath と同一
  total = Σ|pts[k]-pts[k-1]|; if total==0: return pts[0]
  target = total*f; walk segments until target consumed; lerp within segment
```

### 3.4 UpdateSun — `sim_min` → SunSky(昼夜)
```
SimMinAt(T): lerp(Data.sim_min[s0], Data.sim_min[s0+1], f)
UpdateSun(minute):
  day = ((minute%1440)+1440)%1440
  ang = day/1440*2π - π/2                # 6:00 で日の出付近(ビューアと同モデル)
  el  = sin(ang)                          # -1..1(正午最大)
  Sun->SetWorldRotation(FRotator(pitch = -degrees(asin(clamp(el))), yaw = day/1440*360, 0))
  Sun->SetIntensity(lerp(0.2, 8.0 lux換算, max(el,0)))   # or Lumen 前提で Sky と併用
  # 空は SkyAtmosphere + SkyLight を Sun の向きに追従させる(BP_Sky_Sphere/SunSky を使うのが早い)
```

`viz/make_viewer3d.py updateSky()` / `viz/blender_import.py _sun_state()` と **同じ角度・昼度モデル**なので、
3 ビューアで時刻→太陽の見えが揃う。

---

## 4. マテリアル / 色分け(ISM custom data)

- `AgentISM->NumCustomDataFloats = 1`、値 0/1 を「居住/来訪」に割当て。
- マテリアルで `PerInstanceCustomData`(index 0)を取り出し `Lerp(青, 橙, data0)`。
  ビューアの居住=青(#54a0ff)/来訪=橙(#ffb454)に合わせる。
- 職業色など多値にしたい場合は `NumCustomDataFloats = 3`(RGB)にして Python 側で色を焼く。
- 半透明(屋内の人を見る X-ray)は建物側マテリアルの Opacity を切り替える(別 UI)。

---

## 5. Blueprint だけで組む場合の要点(C++ 不要ルート)

1. `BP_ShibuyaSimReplay`(親 Actor)に **AgentISM / CarISM** を 2 つ追加。
2. 変数 `JsonPath`(String)、`T`(Float)、展開済み配列は `Load File to String`(拡張プラグイン
   *Runtime Files*、または `VaRest`/`JsonBlueprintUtilities`(UE5.1+ 標準の低レベル JSON))で読む。
   標準の **`JsonBlueprintUtilities`** で `sim_ue.json` を `FJsonObjectWrapper` に読み込み可能。
3. `Event BeginPlay` で配列展開 + `Add Instance` ×NA/CAR_CAP。
4. `Event Tick` で 3.2 のループ。`Batch Update Instances Transforms`(ISM ノード)で一括更新。
5. 昼夜は `SunSky` アクタの Solar Time を `SimMinAt(T)/60` で毎tick更新でも可(手軽)。

> Blueprint の配列走査は大規模(数千体)で重くなり得る。**1 万体規模を狙うなら C++ 実装 or
> Niagara+VAT へ**(README_UE.md「大量エージェント」参照)。数百〜千体台なら BP で十分実用。

---

## 6. 実機での検証項目(未検証)

- `AddComponentByClass` / `add_component_by_class` の可否(UE バージョン差)。不可なら BP で ISM を持つ親を用意。
- `BatchUpdateInstancesTransforms` の引数順・`bMarkRenderStateDirty` の要否。
- ISM の原点(シリンダ中心 vs 足元)に応じた `+AGENT_H/2` の符号。
- `JsonBlueprintUtilities` で数万要素配列のパース性能(重ければ C++ の `FJsonSerializer`)。
- 進行方向 Yaw の要否(カプセルなら不要、キャラメッシュなら必要)。
