"""Render an atlas of shape keys / head-bone rotations on the rich
character so their meaning and sign can be read before authoring clips.

  Blender --background --python sk_atlas.py -- --blend work/mblab_char_rich.blend \
      --out work/atlas [--keys Expressions_mouthClosed_max,...] [--head pitch=10,yaw=10,roll=10]

Each entry renders one 512x512 still with that key at 1.0 (or that head
rotation in degrees); assemble them with sheet.py.
"""
import os, sys, math, argparse
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_scene as S  # noqa: E402

DEFAULT_KEYS = [
    "Expressions_browOutVertL_max", "Expressions_browOutVertL_min", "Expressions_browsMidVert_max",
    "Expressions_browsMidVert_min", "Expressions_browSqueezeL_max", "Expressions_browSqueezeL_min",
    "Expressions_eyeClosedL_min", "Expressions_eyeSquintL_max", "Expressions_eyesHoriz_max",
    "Expressions_eyesVert_max", "Expressions_mouthClosed_max", "Expressions_mouthClosed_min",
    "Expressions_mouthSmile_min", "Expressions_mouthSmileL_max", "Expressions_mouthHoriz_max",
    "Expressions_mouthHoriz_min", "Expressions_mouthOpenTeethClosed_max", "Expressions_mouthOpenO_max",
    "Expressions_mouthBite_max", "Expressions_mouthLowerOut_max", "Expressions_mouthLowerOut_min",
    "Expressions_mouthInflated_min", "Expressions_cheekSneerL_max", "Expressions_nostrilsExpansion_max",
    "Expressions_chestExpansion_max", "Expressions_eyeClosedPressureL_max",
]

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--keys", default=",".join(DEFAULT_KEYS))
ap.add_argument("--head", default="pitch=10,yaw=10,roll=10")
ap.add_argument("--samples", type=int, default=16)
args = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=args.blend)
scn = bpy.context.scene
body = bpy.data.objects[S.BODY_NAME]; arm = bpy.data.objects[S.ARM_NAME]
kb = body.data.shape_keys.key_blocks
scn.eevee.taa_render_samples = args.samples
os.makedirs(args.out, exist_ok=True)
pb = arm.pose.bones["head"]; pb.rotation_mode = 'XYZ'


def reset():
    for k in kb:
        k.value = 0.0
    pb.rotation_euler = (0, 0, 0)


def shoot(name):
    scn.render.filepath = os.path.join(args.out, name + ".png")
    bpy.ops.render.render(write_still=True)


reset(); shoot("00_rest")
for key in [k for k in args.keys.split(",") if k]:
    if key not in kb:
        print("[atlas] no such key", key); continue
    reset(); kb[key].value = 1.0; shoot(key.replace("Expressions_", ""))
for spec in [h for h in args.head.split(",") if h]:
    axis, deg = spec.split("="); deg = float(deg)
    reset()
    idx = {"pitch": 0, "yaw": 1, "roll": 2}[axis]
    rot = [0.0, 0.0, 0.0]; rot[idx] = math.radians(deg)
    pb.rotation_euler = rot
    shoot(f"head_{axis}_{int(deg)}")
print("[atlas] done ->", args.out)
