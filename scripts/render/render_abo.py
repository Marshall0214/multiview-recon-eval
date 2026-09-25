"""Render an ABO furniture model from candidate / test viewpoints with Blender (Cycles).

Run:  blender -b -P scripts/render/render_abo.py -- --glb X.glb --out DIR [--res 800]

Outputs (DIR):
  images/{split}_{i:03d}.png   RGBA, transparent background
  mesh_gt.ply                  GT mesh in the render frame (metric, bbox centred at origin)
  cameras.json                 intrinsics + per-view c2w (OpenCV convention: x right, y down, z forward)

Viewpoints (README §4.2): directions on a sphere around the object, elevation in
[ELEV_MIN, ELEV_MAX] (an arm cannot look from below the floor), radius = RADIUS_SCALE * bbox diagonal.
  cand: 200 Fibonacci-sphere directions (deterministic, near uniform)
  test:  60 random directions (separate seed, disjoint from cand)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ELEV_MIN, ELEV_MAX = -15.0, 75.0
RADIUS_SCALE = 1.2
FOV_DEG = 45.0
N_CAND, N_TEST = 200, 60


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--res", type=int, default=800)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def elev_ok(d):
    e = math.degrees(math.asin(d[2]))
    return ELEV_MIN <= e <= ELEV_MAX


def fibonacci_dirs(n):
    """n near-uniform directions inside the elevation band (z up)."""
    out, k, total = [], 0, n * 4
    golden = math.pi * (3.0 - math.sqrt(5.0))
    while len(out) < n:
        # oversample the full sphere and keep the band; grow until we have n
        out = []
        for i in range(total):
            z = 1 - 2 * (i + 0.5) / total
            r = math.sqrt(1 - z * z)
            d = (r * math.cos(golden * i), r * math.sin(golden * i), z)
            if elev_ok(d):
                out.append(d)
        total += n
        k += 1
    idx = np.linspace(0, len(out) - 1, n).round().astype(int)
    return [out[i] for i in idx]


def random_dirs(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if elev_ok(v):
            out.append(tuple(v))
    return out


def look_at_c2w(eye, target=(0, 0, 0)):
    """OpenCV-convention camera-to-world (z forward, y down)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:  # looking straight down / up
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def setup_scene(glb, res, samples):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

    # centre the bbox at the origin, keep metric scale
    pts = np.array([(o.matrix_world @ Vector(c))[:] for o in meshes for c in o.bound_box])
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2
    for o in bpy.context.scene.objects:
        if o.parent is None:
            o.location = Vector(o.location) - Vector(centre)
    bpy.context.view_layer.update()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"  # no filmic tone curve: keep colours simple

    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(d.type == backend for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                print(f"[render] using {backend}")
                break
        except TypeError:
            continue

    # static, view-independent-ish lighting: uniform sky + one soft sun
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 2.0
    sun.data.angle = math.radians(30)
    sun.rotation_euler = (math.radians(40), 0, math.radians(30))
    scene.collection.objects.link(sun)

    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.angle = math.radians(FOV_DEG)
    cam.data.clip_start, cam.data.clip_end = 0.01, 100
    scene.collection.objects.link(cam)
    scene.camera = cam
    return meshes, hi - lo


def export_gt_mesh(meshes, path):
    for o in bpy.context.scene.objects:
        o.select_set(o in meshes)
    bpy.ops.wm.ply_export(filepath=str(path), export_selected_objects=True, apply_modifiers=True,
                          export_normals=False, export_uv=False, export_colors="NONE",
                          forward_axis="Y", up_axis="Z")


def main():
    args = parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    meshes, extent = setup_scene(args.glb, args.res, args.samples)
    diag = float(np.linalg.norm(extent))
    radius = RADIUS_SCALE * diag
    export_gt_mesh(meshes, out / "mesh_gt.ply")

    cam = bpy.context.scene.camera
    focal = 0.5 * args.res / math.tan(math.radians(FOV_DEG) / 2)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV -> Blender camera axes

    frames = []
    for split, dirs in (("cand", fibonacci_dirs(N_CAND)), ("test", random_dirs(N_TEST, args.seed + 1))):
        for i, d in enumerate(dirs):
            c2w = look_at_c2w(np.asarray(d) * radius)
            cam.matrix_world = Matrix((c2w @ flip).tolist())
            name = f"{split}_{i:03d}"
            bpy.context.scene.render.filepath = str(out / "images" / f"{name}.png")
            bpy.ops.render.render(write_still=True)
            frames.append({"name": name, "split": split, "c2w": c2w.tolist()})

    meta = {
        "glb": Path(args.glb).name,
        "width": args.res, "height": args.res,
        "fx": focal, "fy": focal, "cx": args.res / 2, "cy": args.res / 2,
        "bbox_extent": extent.tolist(), "bbox_diag": diag, "radius": radius,
        "elev_range_deg": [ELEV_MIN, ELEV_MAX],
        "frames": frames,
    }
    (out / "cameras.json").write_text(json.dumps(meta, indent=1))
    print(f"[render] done: {len(frames)} views -> {out}")


if __name__ == "__main__":
    main()
