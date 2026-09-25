"""Simulated handheld capture to validate the real-data pipeline before real photos exist.

Scene: ABO furniture standing on a textured floor, the printed A4 ArUco page lying flat next to it.
Cameras mimic docs/capture_guide.md: two handheld rings (waist ~30 deg down, knee ~12 deg down),
iPhone-12-main-camera-like FOV, 4:3 frames, small position/aim jitter. Only JPEGs are written
(no poses) -- the pipeline must recover everything with COLMAP + ArUco, exactly as with real data.

Run: blender -b -P scripts/real/render_sim_capture.py -- --glb X.glb --board docs/aruco_board_a4.png --out DIR
GT written to DIR/gt.json (object dims in mm, measured on the GT mesh in the floor frame).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

HFOV_DEG = 65.0   # iPhone 12 main camera (26 mm equiv.) ~ 65 deg horizontal
RES = (1600, 1200)
A4 = (0.210, 0.297)


def args_():
    argv = sys.argv[sys.argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--board", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per_ring", type=int, default=45)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def textured_material(name, scale, c1, c2):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 12
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = scale * 0.6
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 0.6
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = c1
    ramp.color_ramp.elements[1].color = c2
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs[1])
    nt.links.new(vor.outputs["Color"], mix.inputs[2])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def main():
    a = args_()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.glb)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    pts = np.array([(o.matrix_world @ v.co)[:] for o in meshes for v in o.data.vertices])
    lo, hi = pts.min(0), pts.max(0)
    shift = Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))  # stand on the floor, centred
    for o in bpy.context.scene.objects:
        if o.parent is None:
            o.location = Vector(o.location) + shift
    bpy.context.view_layer.update()
    pts = pts + np.array(shift)
    height = float(pts[:, 2].max())
    half = (hi - lo)[:2] / 2

    # GT dims measured the same way as the pipeline does (min-area footprint, height above floor)
    hull_xy = pts[:, :2]
    best = None
    for ang in np.radians(np.arange(0, 90, 0.5)):
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        q = hull_xy @ R
        ext = q.max(0) - q.min(0)
        if best is None or ext[0] * ext[1] < best[0] * best[1]:
            best = ext
    w, d = max(best), min(best)

    # floor + walls-free room: a big textured floor
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.data.materials.append(textured_material("floor", 18.0, (0.55, 0.45, 0.35, 1), (0.9, 0.85, 0.75, 1)))

    # A4 page with the ArUco board, flat on the floor beside the object
    bpy.ops.mesh.primitive_plane_add(size=1, location=(half[0] + 0.25, 0, 0.0005))
    page = bpy.context.active_object
    page.scale = (A4[0], A4[1], 1)
    page.rotation_euler = (0, 0, math.radians(20))
    mat = bpy.data.materials.new("page")
    mat.use_nodes = True
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(Path(a.board).resolve()))
    tex.interpolation = "Closest"
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    page.data.materials.append(mat)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = a.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = RES
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    scene.view_settings.view_transform = "Standard"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = dev.type == "CUDA"
    scene.cycles.device = "GPU"

    world = bpy.data.worlds.new("w")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 2.5
    sun.data.angle = math.radians(40)
    sun.rotation_euler = (math.radians(35), 0, math.radians(60))
    scene.collection.objects.link(sun)

    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.angle = math.radians(HFOV_DEG)
    scene.collection.objects.link(cam)
    scene.camera = cam

    size = float(np.linalg.norm(hi - lo))
    target = Vector((0, 0, height * 0.45))
    k = 0
    for elev, rscale in ((30.0, 1.5), (12.0, 1.3)):
        r = rscale * size
        for i in range(a.per_ring):
            az = 2 * math.pi * i / a.per_ring + rng.normal(0, 0.03)
            e = math.radians(elev + rng.normal(0, 2))
            eye = Vector((r * math.cos(e) * math.cos(az), r * math.cos(e) * math.sin(az),
                          height * 0.45 + r * math.sin(e))) + Vector(rng.normal(0, 0.02, 3).tolist())
            aim = target + Vector(rng.normal(0, 0.03, 3).tolist())
            direction = aim - eye
            cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            cam.location = eye
            scene.render.filepath = str(out / "images" / f"IMG_{k:04d}.jpg")
            bpy.ops.render.render(write_still=True)
            k += 1

    gt = {"glb": Path(a.glb).name, "dims_mm": {"height": height * 1000, "width": w * 1000, "depth": d * 1000},
          "marker_mm": 50.0, "n_images": k}
    (out / "gt.json").write_text(json.dumps(gt, indent=1))
    (out / "measurements.json").write_text(json.dumps({"marker_mm": 50.0, "object": "sim " + gt["glb"],
                                                       "dims_mm": gt["dims_mm"]}, indent=1))
    print("[sim] done", gt)


if __name__ == "__main__":
    main()
