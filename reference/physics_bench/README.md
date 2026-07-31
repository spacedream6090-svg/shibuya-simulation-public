# physics_bench — 物理エンジン選定(P2)の比較検証プロトタイプ

`docs/plans/source/physics-instructions.md` Part P2 の「口頭比較ではなく、同一シナリオを
各候補で実際に動かした実測で評価する」を満たすための使い捨てベンチです。
**`src/` と `conf/` は 1 バイトも変更しません**(`src/society/world/sfm_core.py` と
`indoor_flow.py` は import して読み取り利用するだけ)。

結論と比較表は `docs/research/physics-engine-selection.md` にあります。

---

## 候補

| 候補 | 実体 | 備考 |
|---|---|---|
| (a) SFM | `src/society/world/indoor_flow.WallCrowd`(= `sfm_core.Crowd` + 壁斥力 + 近傍 cap)を薄く包む | 自前資産の再利用。パラメータは Helbing2000 の確定値 |
| (b) ORCA | `orca_min.py`(本ディレクトリの自前最小実装) | **RVO2 の Python バインディングは本機で導入不能**。理由は `orca_min.py` 冒頭の docstring に実測ログ付きで記載 |

### RVO2 バインディングが使えなかった理由(実測)

```
$ python -m pip install rvo2
ERROR: Could not find a version that satisfies the requirement rvo2 (from versions: none)
$ python -m pip install pyrvo2 / python-rvo2 / rvo2-python
ERROR: (同上・PyPI に配布物なし)
$ command -v cl.exe / gcc / cmake   → いずれも未検出
$ python -c "import Cython"         → ModuleNotFoundError
```

既知のバインディング(`sybrenstuvel/Python-RVO2`, `mit-acl/Python-RVO2`)は
Cython + CMake + C++ コンパイラでのソースビルドが必須で、本機にツールチェーンが無い。
そのため ORCA は RVO2 参照実装(`Agent.cpp`)の式をそのまま Python へ移植した
`orca_min.py` で評価しています(移植の忠実さ・意図的に変えた点は同 docstring 参照)。

---

## 再現手順

```powershell
# 依存: numpy(既存) + matplotlib(プロット用)。追加インストール不要なら matplotlib のみ。
cd C:\Users\<user>\Desktop\shibuya-simulation
python -m reference.physics_bench.run_bench --out reference\physics_bench\out
```

* 所要時間: フルで十数分(CPU 依存)。`--quick` で短縮版(約 4 分)。
* `--skip-plots` で PNG 生成を省略。`--plots-only` で既存 `results.json` から図だけ作り直す
  (軌跡用のシナリオだけ再実行するので数十秒)。
* 生成物

| ファイル | 内容 |
|---|---|
| `out/results.json` | 全指標(決定論ハッシュ / スループット / 基本図 / dt 安定性 / 較正感度 / 対向流 / 交差流 / ゲート / エンジン実測メモ) |
| `out/fd_dt0.1.png`, `out/fd_dt0.02.png` | 基本図(密度-速度)+ Weidmann(1993) 重ね描き |
| `out/dt_stability.png` | 前進速度 vs 物理 dt(前進 Euler の安定性) |
| `out/counterflow_sweep.png` | 対向流の流量効率とレーン秩序 vs 密度 |
| `out/traj_counterflow_w3_*.png` / `w6_*` | 対向流の軌跡(赤=右向き / 青=左向き) |
| `out/traj_crossing_*.png` | 4方向交差流の軌跡(ストリーム別に着色) |

### 決定論の確かめ方

1. **同一プロセス内 2 回走**: `results.json` の `determinism.<scenario>.<engine>` に
   `hash_run1` / `hash_run2` / `byte_identical` が入る。
2. **別プロセス間**: 上のコマンドを 2 回実行し、2 つの `results.json` の
   `determinism.*.hash_run1` を比較する(同値なら再現)。

    ```powershell
    python -m reference.physics_bench.run_bench --out out_a --skip-plots
    python -m reference.physics_bench.run_bench --out out_b --skip-plots
    python -c "import json;a=json.load(open('out_a/results.json',encoding='utf-8'));b=json.load(open('out_b/results.json',encoding='utf-8'));print(a['determinism']==b['determinism'])"
    ```

---

## シナリオ

| ID | 内容 | 何を測るか |
|---|---|---|
| A `fd_periodic` | 周期境界の一方向通路 20m × 3m(周期像=ゴーストで厳密な周期性)。密度 0.2〜3.0 /m² | 基本図(密度-速度)。Weidmann(1993) と重ね描き |
| B `counterflow` | 幅員違い通路の対向流 200 体。幅 3m×長 60m / 幅 6m×長 30m(**面密度を 1.11 /m² に揃える**) | レーン形成(方向分離度 φ とその帰無値)・速度・重なり |
| C `crossing` | 開放正方領域 20m × 20m の 4 方向交差流(スクランブル風)200 体・壁なし | 速度効率・最小間隔・重なり |
| D `gate_stitching` | B と同じ通路で流入/流出ゲート規則を `blind` / `guarded` に変えて対照 | 境界縫合性(急停止・振動・瞬間移動・重なり) |

ゲート方式:

* **blind** — 出口を越えた個体を占有チェックなしで即入口へ再投入(P3 の素朴実装)。
* **guarded** — 出口を越えた個体は待機列(遠方の駐機スロット)へ退避。入口候補が空いている個体だけ
  id 昇順で入場させ、**退場時の速度をそのまま復元**する(P3 の「初速をグラフ側から連続に引き継ぐ」に対応)。

## 公平性のために両候補で揃えた条件

* 初期配置・希望速度 `v0` ∈ [1.0,1.4) m/s・半径 `radius` ∈ [0.25,0.35) m
  (`indoor_flow.desired_speed` / `body_radius` = agent_id の blake2b 安定ハッシュ由来)
* 近傍上限 12 体(`indoor_flow.NEIGHBOR_CAP`)、最高速度 = 1.3·v0(`sfm_core.V_MAX_FACTOR`)
* 壁の線分集合、物理 dt、記録方法、指標の計算コード

## 既知の限界

* `orca_min.py` の障害物処理は半平面近似(RVO2 の障害物 ORCA を移植していない)。直線壁の通路では
  ほぼ等価だが、鋭角コーナー・薄い障害物では RVO2 と挙動が変わりうる。
* 基本図の密度は測定矩形内の頭数/面積(古典法)。Voronoi 法(Steffen & Seyfried 2010)は未実装。
* レーン形成の秩序変数は自前定義(横断ビンの方向分離度)。Feliciani & Nishinari (2016) の
  2 次元 order parameter とは別物。有限サイズのベースライン(方向ラベル無作為化)を必ず併記している。
* 実 LLM も本体シミュレータも一切動かさない。ここで測っているのは物理層単体の性能だけ。
