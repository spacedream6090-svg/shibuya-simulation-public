"""Blender 取込みスクリプト(バッチE / 3D 可視化)。

使い方(Blender 4.x):
  blender -b -P viz/blender_import.py -- --scene runs/<name>/scene3d --save out.blend
  # -b: バックグラウンド, -P: このスクリプトを実行, -- 以降がスクリプト引数
オプション:
  --scene DIR    scene.json / tracks.json / buildings.glb のあるディレクトリ(必須)
  --save FILE    出力 .blend パス(既定: <scene>/../shibuya.blend)
  --use-glb      建物を buildings.glb からインポート(既定は scene.json から押出し=建物ごとに命名)
  --turntable    カメラ用 Empty に 360° ターンテーブルのキーフレームを付ける
  --frames-per-step N   1 step(10分)あたりのフレーム数(既定 8)

依存:
  - bpy(Blender 同梱)。pyarrow には一切依存しない(scene.json/tracks.json/buildings.glb のみ読む)。
  - bpy は「スクリプト実行時」に import する(Blender 非搭載環境でもファイル自体はレビュー可能)。
座標系:
  Blender は Z-up・右手系で local-m(X=east,Y=north,Z=up)と一致 → 変換不要でそのまま配置。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

FLOOR_HEIGHT_DEFAULT = 3.5

# 建物 kind → RGB(0..1)。export_3d / viewer3d と同系統。
KIND_RGB = {
    "station": (0.91, 0.42, 0.20),
    "retail": (0.23, 0.66, 0.61),
    "office": (0.36, 0.56, 0.84),
    "residential": (0.48, 0.72, 0.42),
    "hotel": (0.79, 0.42, 0.69),
    "public": (0.61, 0.50, 0.83),
    "generic": (0.72, 0.75, 0.78),
    "house?": (0.66, 0.62, 0.57),
}
DEFAULT_RGB = (0.72, 0.75, 0.78)
VISITOR_RGB = (1.00, 0.71, 0.33)
RESIDENT_RGB = (0.33, 0.63, 1.00)
CAR_RGB = (0.95, 0.69, 0.20)


# ------------------------------------------------------------------ データ読込
def load_scene(scene_dir: Path):
    scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    tracks_p = scene_dir / "tracks.json"
    if tracks_p.exists():
        return scene, json.loads(tracks_p.read_text(encoding="utf-8"))
    # export_3d --no-tracks-json のラン: 量子化バイナリを復号して従来どおりの dict にする
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tracks_bin", Path(__file__).resolve().parents[1] / "scripts" / "tracks_bin.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    tracks = tb.load_tracks(scene_dir)
    if tracks is None:
        raise SystemExit(f"[blender_import] {scene_dir} に tracks.json / tracks.bin が無い")
    return scene, tracks


# ------------------------------------------------------------------ 補間(posAt 移植)
def _path_len(pts):
    L = 0.0
    for i in range(1, len(pts)):
        L += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return L


def along_path(pts, f):
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


def agent_pos(tracks, i, s0, f):
    """viz/make_viewer.py posAt を 1 エージェント分に落とした版。戻り値 [x,y,w]。"""
    P, M, NS = tracks["positions"], tracks["moves"], tracks["meta"]["nSteps"]
    w = P[s0][i][2]
    if w != 0:
        return [P[s0][i][0], P[s0][i][1], w]
    nm = M[s0 + 1][i] if s0 + 1 < NS else None
    if nm:
        p = along_path(nm[1], f)
        return [p[0], p[1], 0]
    a = P[s0][i]
    b = P[min(s0 + 1, NS - 1)][i]
    if b[2] != 0:
        return [a[0], a[1], 0]
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, 0]


def up_of(w, fh):
    if w >= 1000:
        floor = (w - 1000) % 100
        return max((floor - 0.5) * fh, 0.8)
    return 1.2


# ================================================================== Blender 構築
def build(scene_dir: Path, save_path: Path, use_glb: bool,
          turntable: bool, frames_per_step: int) -> None:
    import bpy  # スクリプト実行時に import(Blender 内でのみ有効)

    scene, tracks = load_scene(scene_dir)
    fh = scene.get("meta", {}).get("floor_height", FLOOR_HEIGHT_DEFAULT)
    NS = tracks["meta"]["nSteps"]
    sim_min = tracks["sim_min"]

    # --- まっさらなシーンから
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bscene = bpy.context.scene

    def new_collection(name):
        c = bpy.data.collections.new(name)
        bscene.collection.children.link(c)
        return c

    col_bld = new_collection("buildings")
    col_road = new_collection("roads")
    col_rail = new_collection("rails")
    col_agent = new_collection("agents")

    def make_material(name, rgb, alpha=1.0):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.85
            if alpha < 1.0 and "Alpha" in bsdf.inputs:
                bsdf.inputs["Alpha"].default_value = alpha
        m.diffuse_color = (rgb[0], rgb[1], rgb[2], alpha)
        if alpha < 1.0:
            m.blend_method = 'BLEND'
        return m

    kind_mat = {k: make_material(f"bld_{k}", v) for k, v in KIND_RGB.items()}
    kind_mat["_default"] = make_material("bld_default", DEFAULT_RGB)

    # ------------------------------------------------ 建物
    if use_glb:
        glb = scene_dir / "buildings.glb"
        bpy.ops.import_scene.gltf(filepath=str(glb))
        # インポート結果を buildings コレクションへ寄せる
        for ob in list(bpy.context.selected_objects):
            for c in ob.users_collection:
                c.objects.unlink(ob)
            col_bld.objects.link(ob)
    else:
        for b in scene["buildings"]:
            fp = b["footprint"]
            # 閉環(末尾=先頭)を除去して環頂点に
            ring = fp[:-1] if len(fp) > 1 and fp[0] == fp[-1] else fp
            n = len(ring)
            if n < 3:
                continue
            base = b.get("base", 0.0)
            top = base + max(b.get("height", fh), 1.0)
            verts = [(p[0], p[1], base) for p in ring] + [(p[0], p[1], top) for p in ring]
            faces = [list(range(n - 1, -1, -1)),                 # 底(下向き)
                     list(range(n, 2 * n))]                       # 天(上向き)
            for e in range(n):                                    # 側壁
                a, c2 = e, (e + 1) % n
                faces.append([a, c2, c2 + n, a + n])
            mesh = bpy.data.meshes.new(b["id"])
            mesh.from_pydata(verts, [], faces)
            mesh.update()
            ob = bpy.data.objects.new(b.get("name") or b["id"], mesh)
            ob.data.materials.append(kind_mat.get(b.get("kind"), kind_mat["_default"]))
            col_bld.objects.link(ob)

    # ------------------------------------------------ 道路・線路(カーブ)
    def add_curve(coll, name, poly_xyz, bevel, rgb):
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        cu.bevel_depth = bevel
        sp = cu.splines.new('POLY')
        sp.points.add(len(poly_xyz) - 1)
        for i, (x, y, z) in enumerate(poly_xyz):
            sp.points[i].co = (x, y, z, 1.0)
        ob = bpy.data.objects.new(name, cu)
        ob.data.materials.append(rgb)
        coll.objects.link(ob)
        return ob

    road_mat = make_material("road", (0.29, 0.33, 0.39))
    rail_mat = make_material("rail", (0.60, 0.64, 0.70))
    subway_mat = make_material("subway", (0.54, 0.36, 0.94), alpha=0.4)

    for j, r in enumerate(scene["roads"]):
        g = r["g"]
        if len(g) < 2:
            continue
        add_curve(col_road, f"road_{j}", [(x, y, 0.3) for x, y in g], 0.7, road_mat)
    for j, r in enumerate(scene["rails"]):
        g = r["g"]
        if len(g) < 2:
            continue
        z = r.get("z", 0.0)
        if r.get("kind") == "subway":
            add_curve(col_rail, r.get("name") or f"subway_{j}",
                      [(x, y, z) for x, y in g], 3.0, subway_mat)
        else:
            add_curve(col_rail, r.get("name") or f"rail_{j}",
                      [(x, y, z + 0.6) for x, y in g], 1.0, rail_mat)

    # ------------------------------------------------ 地面
    bpy.ops.mesh.primitive_plane_add(size=6000, location=(0, 0, -0.05))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(make_material("ground", (0.08, 0.10, 0.13)))

    # ------------------------------------------------ エージェント(カプセル)
    cap_mesh = _capsule_mesh(bpy, r=2.6, h=5.0, seg=12, rings=3)
    visitor_mat = make_material("agent_visitor", VISITOR_RGB)
    resident_mat = make_material("agent_resident", RESIDENT_RGB)
    # メッシュを共有しつつ色をオブジェクト単位で変えるためスロットを 1 つ用意
    cap_mesh.materials.append(resident_mat)

    fps = 24
    bscene.render.fps = fps
    bscene.frame_start = 1
    bscene.frame_end = 1 + NS * frames_per_step

    agents_meta = tracks["agents"]
    for i, am in enumerate(agents_meta):
        ob = bpy.data.objects.new(am.get("name") or f"agent{am.get('id', i)}", cap_mesh)
        col_agent.objects.link(ob)
        # material_slots[0].link='OBJECT' で「メッシュ共有・色は個体別」を実現
        ob.material_slots[0].link = 'OBJECT'
        ob.material_slots[0].material = visitor_mat if am.get("visitor") else resident_mat
        _keyframe_agent(ob, tracks, i, fh, frames_per_step)

    # ------------------------------------------------ 車(traffic)。全車同色なのでメッシュ共有で可
    car_mesh = _box_mesh(bpy, 4.5, 2.4, 2.2)
    car_mesh.materials.append(make_material("car", CAR_RGB))
    max_cars = min(120, max((len(t.get("segs", [])) for t in tracks["traffic"]), default=0))
    for ci in range(max_cars):
        ob = bpy.data.objects.new(f"car{ci}", car_mesh)
        col_agent.objects.link(ob)
        _keyframe_car(ob, tracks, ci, frames_per_step)

    # ------------------------------------------------ 太陽(日周)+ 空
    sun_data = bpy.data.lights.new("Sun", 'SUN')
    sun = bpy.data.objects.new("Sun", sun_data)
    bscene.collection.objects.link(sun)
    _keyframe_sun(bpy, sun, sun_data, sim_min, NS, frames_per_step)
    _keyframe_world(bpy, bscene, sim_min, NS, frames_per_step)

    # ------------------------------------------------ カメラ + オービット Empty
    empty = bpy.data.objects.new("orbit", None)
    empty.empty_display_size = 40
    bscene.collection.objects.link(empty)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 32
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (360, -520, 420)
    bscene.collection.objects.link(cam)
    cam.parent = empty
    con = cam.constraints.new('TRACK_TO')
    con.target = empty
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    bscene.camera = cam
    if turntable:
        empty.rotation_euler = (0, 0, 0)
        empty.keyframe_insert("rotation_euler", frame=bscene.frame_start)
        empty.rotation_euler = (0, 0, 2 * math.pi)
        empty.keyframe_insert("rotation_euler", frame=bscene.frame_end)

    # ------------------------------------------------ レンダ設定(EEVEE)
    try:
        bscene.render.engine = 'BLENDER_EEVEE_NEXT'   # Blender 4.2+
    except TypeError:
        bscene.render.engine = 'BLENDER_EEVEE'        # 4.0 / 4.1
    bscene.render.resolution_x = 1920
    bscene.render.resolution_y = 1080
    bscene.render.film_transparent = False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
    print(f"[blender_import] saved: {save_path}  "
          f"(buildings={len(scene['buildings'])}, agents={len(agents_meta)}, "
          f"cars={max_cars}, frames={bscene.frame_end})")


# ------------------------------------------------------------------ メッシュ生成
def _capsule_mesh(bpy, r, h, seg=12, rings=3):
    """半径 r・円柱部高さ h のカプセルを from_pydata で解析生成(bmesh 不要)。"""
    verts, faces = [], []
    half = h / 2.0
    lat = []  # (z, radius) の輪。下極→上極。
    # 下半球
    for k in range(rings + 1):
        a = -math.pi / 2 + (math.pi / 2) * (k / rings)
        lat.append((-half + math.sin(a) * r, math.cos(a) * r))
    # 上半球
    for k in range(1, rings + 1):
        a = (math.pi / 2) * (k / rings)
        lat.append((half + math.sin(a) * r, math.cos(a) * r))
    # 各輪を seg 分割した頂点
    ring_start = []
    for (z, rad) in lat:
        ring_start.append(len(verts))
        if rad < 1e-6:  # 極
            verts.append((0.0, 0.0, z))
        else:
            for s in range(seg):
                th = 2 * math.pi * s / seg
                verts.append((math.cos(th) * rad, math.sin(th) * rad, z))
    # 輪同士を面で接続
    for li in range(len(lat) - 1):
        z0, r0 = lat[li]
        z1, r1 = lat[li + 1]
        b0, b1 = ring_start[li], ring_start[li + 1]
        if r0 < 1e-6:                       # 下極 → 三角ファン
            for s in range(seg):
                faces.append([b0, b1 + s, b1 + (s + 1) % seg])
        elif r1 < 1e-6:                     # 上極 → 三角ファン
            for s in range(seg):
                faces.append([b1, b0 + (s + 1) % seg, b0 + s])
        else:                              # 帯 → クアッド
            for s in range(seg):
                s2 = (s + 1) % seg
                faces.append([b0 + s, b0 + s2, b1 + s2, b1 + s])
    mesh = bpy.data.meshes.new("capsule")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _box_mesh(bpy, x, y, z):
    hx, hy, hz = x / 2, y / 2, z / 2
    verts = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0),
             (-hx, -hy, z), (hx, -hy, z), (hx, hy, z), (-hx, hy, z)]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
             [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    mesh = bpy.data.meshes.new("car")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


# ------------------------------------------------------------------ キーフレーム
def _linearize(ob):
    ad = ob.animation_data
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'


def _keyframe_agent(ob, tracks, i, fh, fps_step):
    NS = tracks["meta"]["nSteps"]
    P = tracks["positions"]
    M = tracks["moves"]
    prev_hidden = None
    for s in range(NS):
        base_f = 1 + s * fps_step
        w = P[s][i][2]
        hidden = (w == -1 or w == -2)
        # 可視状態が変わったときだけ scale をキー(キー数削減)
        if hidden != prev_hidden:
            ob.scale = (0.0, 0.0, 0.0) if hidden else (1.0, 1.0, 1.0)
            ob.keyframe_insert("scale", frame=base_f)
            prev_hidden = hidden
        if hidden:
            continue
        x, y, _w = P[s][i]
        ob.location = (x, y, up_of(w, fh))
        ob.keyframe_insert("location", frame=base_f)
        # 歩行中(次 step に move ポリライン)はサブキーで道なりに
        nm = M[s + 1][i] if s + 1 < NS else None
        if nm:
            for kf in range(1, fps_step):
                p = agent_pos(tracks, i, s, kf / fps_step)
                ob.location = (p[0], p[1], up_of(p[2], fh))
                ob.keyframe_insert("location", frame=base_f + kf)
    _linearize(ob)


def _keyframe_car(ob, tracks, ci, fps_step):
    NS = tracks["meta"]["nSteps"]
    prev_active = None
    for s in range(NS):
        base_f = 1 + s * fps_step
        segs = tracks["traffic"][s].get("segs", [])
        active = ci < len(segs)
        if active != prev_active:
            ob.scale = (1.0, 1.0, 1.0) if active else (0.0, 0.0, 0.0)
            ob.keyframe_insert("scale", frame=base_f)
            prev_active = active
        if not active:
            continue
        seg = segs[ci]
        for kf in range(0, fps_step):
            p = along_path(seg, kf / fps_step)
            ob.location = (p[0], p[1], 0.9)
            ob.keyframe_insert("location", frame=base_f + kf)
    _linearize(ob)


def _sun_state(minute):
    """sim 分 → (太陽仰角ラジアン, 方位ラジアン, 昼度 0..1)。viewer3d と同モデル。"""
    day = ((minute % 1440) + 1440) % 1440
    ang = (day / 1440) * math.pi * 2 - math.pi / 2
    el = math.sin(ang)                    # -1..1(正午で最大)
    az = (day / 1440) * math.pi * 2
    alt = math.asin(max(-1.0, min(1.0, el)))
    daylight = max(0.0, el)
    return alt, az, daylight


def _keyframe_sun(bpy, sun, sun_data, sim_min, NS, fps_step):
    for s in range(NS):
        f = 1 + s * fps_step
        alt, az, daylight = _sun_state(sim_min[s])
        tilt = math.pi / 2 - max(alt, 0.0)   # 0=天頂, π/2=地平線
        sun.rotation_euler = (tilt, 0.0, az)
        sun.keyframe_insert("rotation_euler", frame=f)
        sun_data.energy = 0.2 + daylight * 4.0
        # 地平線付近は暖色、正午は白
        warm = min(1.0, daylight * 1.8)
        sun_data.color = (1.0, 0.55 + 0.45 * warm, 0.30 + 0.70 * warm)
        sun_data.keyframe_insert("energy", frame=f)
        sun_data.keyframe_insert("color", frame=f)


def _keyframe_world(bpy, bscene, sim_min, NS, fps_step):
    world = bpy.data.worlds.new("sky")
    bscene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if not bg:
        return
    col = bg.inputs["Color"]
    night = (0.02, 0.03, 0.06)
    day = (0.42, 0.58, 0.78)
    dusk = (0.55, 0.30, 0.17)
    for s in range(NS):
        f = 1 + s * fps_step
        _alt, _az, daylight = _sun_state(sim_min[s])
        el = math.sin(((sim_min[s] % 1440) / 1440) * math.pi * 2 - math.pi / 2)
        tw = max(0.0, 1 - abs(el) * 3.2) * 0.55
        def mix(a, b, t):
            return tuple(a[k] + (b[k] - a[k]) * t for k in range(3))
        c = mix(night, day, min(1.0, daylight * 1.6))
        c = mix(c, dusk, tw)
        col.default_value = (c[0], c[1], c[2], 1.0)
        col.keyframe_insert("default_value", frame=f)


# ================================================================== CLI
def parse_args(argv):
    ap = argparse.ArgumentParser(description="Blender importer for shibuya-sim 3D scenes")
    ap.add_argument("--scene", required=True, help="scene3d ディレクトリ")
    ap.add_argument("--save", default=None, help="出力 .blend")
    ap.add_argument("--use-glb", action="store_true", help="建物を glb からインポート")
    ap.add_argument("--turntable", action="store_true", help="カメラをターンテーブル回転")
    ap.add_argument("--frames-per-step", type=int, default=8)
    return ap.parse_args(argv)


def main():
    # Blender では `-- 以降` がスクリプト引数。単体実行時は通常の argv。
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]
    args = parse_args(argv)
    scene_dir = Path(args.scene).resolve()
    save = Path(args.save).resolve() if args.save else (scene_dir.parent / "shibuya.blend")
    build(scene_dir, save, args.use_glb, args.turntable, args.frames_per_step)


if __name__ == "__main__":
    main()
