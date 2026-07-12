"""渋谷シミュ → Unreal Engine 5 取り込み(バッチE / UE 可視化)。

  ★ このスクリプトは *UE5 エディタ内の Python*(Editor Scripting Utilities)で動く。
    この開発環境には UE が無いため **未実行=レビュー用**。実機での検証項目は
    末尾 "検証チェックリスト" と viz/unreal/README_UE.md を参照。`import unreal` は
    UE 内でのみ解決するので、UE 外での構文チェック(python -m py_compile)は
    `unreal` を import しない限り通る(遅延 import 済み)。

やること:
  runs/<name>/scene3d/sim_ue.json(= scripts/export_ue.py が出力、既に UE ワールド座標)を読み、
    (a) エージェント = Instanced Static Mesh(カプセル/シリンダ)の 1 アクタ
    (b) 車       = 別 ISM(ボックス)の同アクタ内コンポーネント
    (c) 軌跡     = 2 方式で取り込み:
        mode="ism"     … ISM + sim_ue.json を Content にコピー。ランタイムの
                          SimReplayActor(viz/unreal/SimReplayActor_DESIGN.md)が毎tick補間再生。
                          80〜1万体レンジはこちらを推奨(Sequencer は ISM インスタンス個別キー不可)。
        mode="sequence"… 小規模(<=~300体)向け。エージェント 1 体 = 1 StaticMeshActor を並べ、
                          Level Sequence に per-step トランスフォームキーをベイク。映像書き出し用。

UE 内での使い方(Output Log の Python コンソール、または Tools>Execute Python Script):
  import sys; sys.path.append(r"C:/Users/.../shibuya-simulation/viz/unreal")
  import import_shibuya_sim as imp
  imp.run(r"C:/Users/.../shibuya-simulation/runs/b_check1/scene3d")            # ISM 方式
  imp.run(r"C:/.../runs/b_check1/scene3d", mode="sequence", max_seq_actors=300) # Sequencer 方式

座標系: sim_ue.json は既に UE(cm, Z-up, 左手系)。ここでは *一切* 座標変換しない
        (変換は scripts/export_ue.py に一本化)。位置合わせ(heading/y_flip/offset)は
        export_ue.py の引数で調整して再出力する運用。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ------------------------------------------------------------------ 定数
CONTENT_DIR = "/Game/ShibuyaSim"          # Content 内の配置先
CAPSULE_ASSET = "/Engine/BasicShapes/Cylinder.Cylinder"  # 人=シリンダ(依存ゼロ)
CAR_ASSET = "/Engine/BasicShapes/Cube.Cube"              # 車=キューブ
AGENT_MAT = "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"

# 人体・車体の実寸(uu=cm)。エンジン基本形状は 100uu 基準なのでスケールで合わせる。
AGENT_DIAM_UU = 55.0     # 直径 ~55cm
AGENT_HEIGHT_UU = 175.0  # 身長 ~175cm
CAR_L, CAR_W, CAR_H = 450.0, 200.0, 150.0

# 状態コード(sim_ue.json / export_ue と共通)
ST_STREET, ST_OFFAREA, ST_SLEEP = 0, -1, -2

# 車 ISM の上限(viewer3d/blender と同系統)
CAR_CAP = 220


# ------------------------------------------------------------------ データ
def load_sim_ue(scene_dir: str) -> dict:
    p = Path(scene_dir) / "sim_ue.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} が無い。先に `python scripts/export_ue.py runs/<name>` を実行して "
            "sim_ue.json を生成してください(座標変換はそちらが担当)。")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 補間(export_ue と対)
def _path_len(pts):
    import math
    L = 0.0
    for i in range(1, len(pts)):
        L += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return L


def _along(pts, f):
    import math
    total = _path_len(pts)
    if total == 0:
        return pts[0]
    target = total * f
    for i in range(1, len(pts)):
        seg = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if target <= seg:
            g = target / seg if seg else 0.0
            return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * g,
                    pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * g]
        target -= seg
    return pts[-1]


def agent_xyz(sim, i, s0, f):
    """viewer3d.posAt の UE 版。戻り値 (ux,uy,uz,state)。移動中は道なり補間。"""
    P, M = sim["positions"], sim["moves"]
    NS = sim["meta"]["nSteps"]
    ux, uy, uz, w = P[s0][i]
    if w != ST_STREET:                         # 屋内/範囲外/睡眠は補間しない
        return ux, uy, uz, w
    nm = M[s0 + 1][i] if s0 + 1 < NS else None
    if nm:                                     # 次stepに移動ポリライン → 道なり
        px, py = _along(nm[1], f)
        return px, py, uz, ST_STREET
    b = P[min(s0 + 1, NS - 1)][i]
    if b[3] != ST_STREET:
        return ux, uy, uz, ST_STREET
    return ux + (b[0] - ux) * f, uy + (b[1] - uy) * f, uz, ST_STREET


# ================================================================== UE 構築
def _subsys(name):
    import unreal
    return unreal.get_editor_subsystem(getattr(unreal, name))


def ensure_meshes():
    """人=シリンダ・車=キューブ・簡易マテリアルをロード(存在すればそれを使う)。"""
    import unreal
    cap = unreal.EditorAssetLibrary.load_asset(CAPSULE_ASSET)
    car = unreal.EditorAssetLibrary.load_asset(CAR_ASSET)
    mat = unreal.EditorAssetLibrary.load_asset(AGENT_MAT)
    if cap is None or car is None:
        raise RuntimeError("Engine の BasicShapes が見つからない。UE5 標準アセット。")
    return cap, car, mat


def _spawn_empty_actor(name, location):
    import unreal
    actors = _subsys("EditorActorSubsystem")
    loc = unreal.Vector(location[0], location[1], location[2])
    actor = actors.spawn_actor_from_class(unreal.Actor, loc, unreal.Rotator(0, 0, 0))
    actor.set_actor_label(name)
    return actor


def _add_ism(actor, mesh, mat):
    """アクタに InstancedStaticMeshComponent を追加してメッシュ/マテリアルを設定。
    add_component_by_class は UE5 で公開(バージョンにより差異あり→README に代替BP手順)。"""
    import unreal
    ism = actor.add_component_by_class(
        unreal.InstancedStaticMeshComponent, False, unreal.Transform(), False)
    ism.set_static_mesh(mesh)
    if mat is not None:
        ism.set_material(0, mat)
    # 半透明/色分けのため per-instance custom data を 1 つ(0=居住,1=来訪)
    ism.set_editor_property("num_custom_data_floats", 1)
    return ism


def _inst_transform(x, y, z, sx, sy, sz):
    import unreal
    return unreal.Transform(
        location=unreal.Vector(x, y, z),
        rotation=unreal.Rotator(0, 0, 0),
        scale=unreal.Vector(sx, sy, sz))


# ------------------------------------------------------------------ 方式1: ISM + DataAsset
def setup_ism_actor(sim, scene_dir):
    """(a)(b) ISM を作り step0 の姿勢で初期配置、sim_ue.json を Content にコピー。
    ランタイム再生は SimReplayActor(BP/C++)が sim_ue.json を読んで駆動する。"""
    import unreal
    cap, car, mat = ensure_meshes()

    actor = _spawn_empty_actor("ShibuyaSimReplay", (0.0, 0.0, 0.0))
    agent_ism = _add_ism(actor, cap, mat)
    car_ism = _add_ism(actor, car, mat)

    # 人のスケール(シリンダは径100uu×高100uu想定 → 実寸へ)
    asx = AGENT_DIAM_UU / 100.0
    asz = AGENT_HEIGHT_UU / 100.0
    NA = len(sim["ids"])
    agents_meta = sim["agents"]
    for i in range(NA):
        ux, uy, uz, w = sim["positions"][0][i]
        hidden = w in (ST_OFFAREA, ST_SLEEP)
        # シリンダは原点が中心 → 足元 z=0 に立たせるため +高さ/2
        t = _inst_transform(ux, uy, (0.0 if hidden else uz + AGENT_HEIGHT_UU / 2),
                            0 if hidden else asx, 0 if hidden else asx,
                            0 if hidden else asz)
        idx = agent_ism.add_instance(t)
        visitor = 1.0 if agents_meta[i].get("visitor") else 0.0
        agent_ism.set_custom_data_value(idx, 0, visitor)

    # 車 ISM を CAR_CAP 個ぶん確保(初期は原点下に隠す)
    csx, csy, csz = CAR_L / 100.0, CAR_W / 100.0, CAR_H / 100.0
    for _ in range(CAR_CAP):
        car_ism.add_instance(_inst_transform(0, 0, -10000, csx, csy, csz))

    # sim_ue.json を Content にコピー(SimReplayActor が実行時に読む)
    dst = _copy_json_to_content(scene_dir)
    unreal.log(f"[shibuya] ISM 構築完了: agents={NA}, car_cap={CAR_CAP}. "
               f"json→{dst}. SimReplayActor に JsonPath='{dst}' を設定して再生。")
    return actor


def _copy_json_to_content(scene_dir):
    """sim_ue.json をプロジェクトの Content/ShibuyaSim/ 直下へコピーして絶対パスを返す。
    (UAsset 化せず生 JSON。SimReplayActor は FFileHelper::LoadFileToString で読む設計)。"""
    import unreal
    proj_content = Path(unreal.Paths.project_content_dir())
    dst_dir = proj_content / "ShibuyaSim"
    dst_dir.mkdir(parents=True, exist_ok=True)
    src = Path(scene_dir) / "sim_ue.json"
    dst = dst_dir / "sim_ue.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return str(dst)


# ------------------------------------------------------------------ 方式2: Level Sequence
def bake_level_sequence(sim, max_seq_actors=300, frames_per_step=8, fps=24):
    """小規模向け。エージェント 1 体=1 StaticMeshActor を並べ、Level Sequence に
    per-step トランスフォームキーをベイク。ISM は Sequencer で個別キー不可のため
    この方式のみアクタ実体を持つ。max_seq_actors を超える体数は間引く。"""
    import unreal
    cap, _car, mat = ensure_meshes()
    NA = min(len(sim["ids"]), max_seq_actors)
    NS = sim["meta"]["nSteps"]
    asx = AGENT_DIAM_UU / 100.0
    asz = AGENT_HEIGHT_UU / 100.0

    # LevelSequence アセット生成
    seq = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        "ShibuyaSimSeq", CONTENT_DIR, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    seq.set_display_rate(unreal.FrameRate(fps, 1))
    seq.set_playback_end(int((NS - 1) * frames_per_step))

    actors = _subsys("EditorActorSubsystem")
    for i in range(NA):
        ux, uy, uz, _w = sim["positions"][0][i]
        sm_actor = actors.spawn_actor_from_object(
            cap, unreal.Vector(ux, uy, uz + AGENT_HEIGHT_UU / 2))
        sm_actor.set_actor_scale3d(unreal.Vector(asx, asx, asz))
        sm_actor.set_actor_label(f"agent_{sim['ids'][i]}")
        comp = sm_actor.static_mesh_component
        if mat is not None:
            comp.set_material(0, mat)

        binding = seq.add_possessable(sm_actor)
        track = binding.add_track(unreal.MovieScene3DTransformTrack)
        section = track.add_section()
        section.set_start_frame(0)
        section.set_end_frame(int((NS - 1) * frames_per_step))
        _bake_agent_keys(section, sim, i, frames_per_step)

    unreal.log(f"[shibuya] Level Sequence '{CONTENT_DIR}/ShibuyaSimSeq' に "
               f"{NA} 体をベイク(<= max_seq_actors={max_seq_actors})。")
    return seq


def _bake_agent_keys(section, sim, i, fps_step):
    """1 体分の Location(X,Y,Z)チャンネルに per-step + 歩行サブキーを打つ。"""
    import unreal
    NS = sim["meta"]["nSteps"]
    chans = section.get_channels()   # [locX, locY, locZ, rotX.., scaleX..]
    cx, cy, cz = chans[0], chans[1], chans[2]

    def key(frame, x, y, z):
        fr = unreal.FrameNumber(int(frame))
        cx.add_key(fr, float(x), interpolation=unreal.MovieSceneKeyInterpolation.LINEAR)
        cy.add_key(fr, float(y), interpolation=unreal.MovieSceneKeyInterpolation.LINEAR)
        cz.add_key(fr, float(z), interpolation=unreal.MovieSceneKeyInterpolation.LINEAR)

    for s in range(NS):
        base_f = s * fps_step
        ux, uy, uz, w = sim["positions"][s][i]
        if w in (ST_OFFAREA, ST_SLEEP):
            continue  # 隠す挙動は簡略化(必要なら visibility track を別途)
        key(base_f, ux, uy, uz + AGENT_HEIGHT_UU / 2)
        nm = sim["moves"][s + 1][i] if s + 1 < NS else None
        if nm:
            for kf in range(1, fps_step):
                px, py, pz, _ = agent_xyz(sim, i, s, kf / fps_step)
                key(base_f + kf, px, py, pz + AGENT_HEIGHT_UU / 2)


# ------------------------------------------------------------------ エントリ
def run(scene_dir, mode="ism", max_seq_actors=300, frames_per_step=8):
    """UE 内から呼ぶ主エントリ。mode: 'ism'(既定/推奨) | 'sequence'(小規模映像)。"""
    sim = load_sim_ue(scene_dir)
    _check_meta(sim)
    if mode == "ism":
        return setup_ism_actor(sim, scene_dir)
    if mode == "sequence":
        return bake_level_sequence(sim, max_seq_actors, frames_per_step)
    raise ValueError(f"unknown mode: {mode}")


def _check_meta(sim):
    """UE 外でも呼べる軽い健全性チェック(unreal 非依存)。"""
    m = sim.get("meta", {})
    assert m.get("units") == "cm", "sim_ue.json は cm 単位のはず(export_ue.py)"
    assert m.get("up_axis") == "Z", "Z-up 前提"
    assert "nSteps" in m and sim.get("positions"), "positions が空"
    return True


# UE の Execute Python Script はモジュール実行になる。環境変数で scene を渡す運用も可。
if __name__ == "__main__":
    _sd = os.environ.get("SHIBUYA_SCENE_DIR")
    _mode = os.environ.get("SHIBUYA_MODE", "ism")
    if _sd:
        run(_sd, mode=_mode)
    else:
        print(__doc__)
