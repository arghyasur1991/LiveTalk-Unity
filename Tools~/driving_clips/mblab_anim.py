"""Set up camera / lights / animation on the finalized MB-Lab character and
render.  Usage:
  Blender --background --python mblab_anim.py -- still 68        (one frame)
  Blender --background --python mblab_anim.py -- anim            (0..124)
"""
import bpy, sys, math
from mathutils import Vector, Euler

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["still", "68"]
mode = argv[0]

bpy.ops.wm.open_mainfile(filepath="/tmp/lp_driver_test/mblab_char.blend")
scn = bpy.context.scene
body = bpy.data.objects["LP_body"]
arm = bpy.data.objects["LP_armature"]

# humanoid_library.blend ships absolute paths into a 4.1 addons dir for the
# iris maps; repoint them at the installed 4.5 copy (otherwise eyes render
# as the "texture missing" pink -> green after AgX).
import os
TEX = os.path.expanduser("~/Library/Application Support/Blender/4.5/scripts/addons/MB-Lab/data/textures")
for img in bpy.data.images:
    if not img.has_data and img.source == 'FILE':
        cand = os.path.join(TEX, os.path.basename(img.filepath))
        if os.path.isfile(cand):
            img.filepath = cand
            img.reload()
            print("repathed", img.name, "->", cand, "size", tuple(img.size))

# ---------------------------------------------------------------- head box
head_bone = arm.data.bones["head"]
neck_bone = arm.data.bones["neck"]
neck_z = (arm.matrix_world @ neck_bone.head_local).z
head_top = (arm.matrix_world @ head_bone.tail_local).z
mw = body.matrix_world
verts = [mw @ v.co for v in body.data.vertices]
face_verts = [v for v in verts if v.z > neck_z + 0.02]
xs = [v.x for v in face_verts]; ys = [v.y for v in face_verts]; zs = [v.z for v in face_verts]
minv = Vector((min(xs), min(ys), min(zs))); maxv = Vector((max(xs), max(ys), max(zs)))
head_h = maxv.z - minv.z
print("head box", tuple(round(c, 3) for c in minv), tuple(round(c, 3) for c in maxv), "h=%.3f" % head_h)
# character faces -Y.  Face centre a bit below the head-box centre (nose).
face_center = Vector(((minv.x + maxv.x) / 2, minv.y, minv.z + head_h * 0.47))
face_front_y = minv.y

# ---------------------------------------------------------------- camera
# frame height at the face plane so that chin->crown (head_h) is ~85 % of the
# frame -> chin-to-hairline face ~60 % (matches the stock talk-neutral framing)
frame_h = head_h / 0.85
focal = 85.0
sensor = 36.0
dist = frame_h * focal / sensor
cam_data = bpy.data.cameras.new("LPCam")
cam_data.lens = focal
cam_data.sensor_width = sensor
cam_data.sensor_fit = 'VERTICAL'
cam_data.sensor_height = sensor
cam_data.clip_start = 0.05
cam = bpy.data.objects.new("LPCam", cam_data)
scn.collection.objects.link(cam)
cam.location = Vector((face_center.x, face_front_y - dist, face_center.z))
cam.rotation_euler = Euler((math.radians(90), 0, 0), 'XYZ')  # look along +Y
scn.camera = cam
print("camera at", tuple(round(c, 3) for c in cam.location), "dist %.3f frame_h %.3f" % (dist, frame_h))

# ---------------------------------------------------------------- lights (soft 3-point)
def area(name, loc, target, energy, size, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, 'AREA')
    d.energy = energy; d.size = size; d.color = color
    o = bpy.data.objects.new(name, d)
    scn.collection.objects.link(o)
    o.location = loc
    direction = Vector(target) - Vector(loc)
    o.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    return o

fc = face_center
area("Key",  (fc.x + 0.9, fc.y - 1.1, fc.z + 0.7), fc, 60, 1.0, (1.0, 0.96, 0.92))
area("Fill", (fc.x - 1.1, fc.y - 1.0, fc.z + 0.2), fc, 25, 1.5, (0.92, 0.95, 1.0))
area("Rim",  (fc.x - 0.4, fc.y + 0.9, fc.z + 0.8), fc, 30, 0.6)

# neutral grey backdrop: world colour + a plane far behind the head
world = bpy.data.worlds.new("LPWorld"); scn.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.32, 0.32, 0.33, 1.0)
bg.inputs[1].default_value = 0.6
bpy.ops.mesh.primitive_plane_add(size=6, location=(fc.x, fc.y + 2.5, fc.z))
plane = bpy.context.active_object; plane.name = "Backdrop"
plane.rotation_euler = Euler((math.radians(90), 0, 0), 'XYZ')
pm = bpy.data.materials.new("BackdropGrey"); pm.use_nodes = True
pm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.36, 0.36, 0.37, 1)
pm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
plane.data.materials.append(pm)

# ---------------------------------------------------------------- render settings
scn.render.engine = 'BLENDER_EEVEE_NEXT'
scn.render.resolution_x = 512
scn.render.resolution_y = 512
scn.render.resolution_percentage = 100
scn.render.fps = 25
scn.frame_start = 0
scn.frame_end = 124
scn.render.image_settings.file_format = 'PNG'
scn.render.image_settings.color_mode = 'RGB'
scn.render.film_transparent = False
scn.eevee.taa_render_samples = 32
scn.eevee.use_shadows = True
scn.eevee.use_raytracing = False
scn.view_settings.view_transform = 'AgX'
scn.view_settings.look = 'None'

# ---------------------------------------------------------------- animation
kb = body.data.shape_keys.key_blocks
for k in kb:
    k.value = 0.0

def comb_to_sk(comb):
    """MB-Lab combined expression -> {shapekey_name: value} (expressionengine rule)."""
    out = {}
    for name, v in comb.items():
        if v < 0.5:
            out[f"{name}_min"] = (0.5 - v) * 2
        else:
            out[f"{name}_max"] = (v - 0.5) * 2
    return {k: v for k, v in out.items() if k in kb and v != 0}

HAPPY01 = {'Expressions_cheekSneerL': 0.55, 'Expressions_browOutVertR': 0.71, 'Expressions_cheekSneerR': 0.55,
           'Expressions_eyeSquintL': 0.7, 'Expressions_mouthSmile': 1.0, 'Expressions_eyeSquintR': 0.7,
           'Expressions_mouthSmileL': 0.59}
SMILE = comb_to_sk(HAPPY01)
BLINK = {"Expressions_eyeClosedL_max": 1.0, "Expressions_eyeClosedR_max": 1.0}
BROWS = {"Expressions_browOutVertL_max": 0.9, "Expressions_browOutVertR_max": 0.9, "Expressions_browsMidVert_max": 0.6}
print("SMILE keys:", SMILE)

def key_sk(names_vals, frame, scale):
    for n, v in names_vals.items():
        kb[n].value = v * scale
        kb[n].keyframe_insert("value", frame=frame)

# blink 40-48: fast close, slower open
key_sk(BLINK, 40, 0.0); key_sk(BLINK, 43, 1.0); key_sk(BLINK, 48, 0.0)
# smile 55-80 to 70 % and release
key_sk(SMILE, 55, 0.0); key_sk(SMILE, 68, 0.7); key_sk(SMILE, 80, 0.0)
# brows 85-100 raise, hold, 105-124 back to neutral
key_sk(BROWS, 85, 0.0); key_sk(BROWS, 100, 1.0); key_sk(BROWS, 105, 1.0); key_sk(BROWS, 124, 0.0)

# head: yaw 15-35 (~8 deg to the character's right and back), tilt 85-124
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones["head"]
pb.rotation_mode = 'XYZ'
# bone local axes: Y runs along the bone (up for the head) -> yaw is local Y,
# roll (tilt toward a shoulder) is local Z.
def key_head(frame, yaw_deg, roll_deg):
    pb.rotation_euler = Euler((0.0, math.radians(yaw_deg), math.radians(roll_deg)), 'XYZ')
    pb.keyframe_insert("rotation_euler", frame=frame)
key_head(0, 0, 0)
key_head(15, 0, 0); key_head(25, -8, 0); key_head(35, 0, 0)
key_head(85, 0, 0); key_head(100, 0, 5); key_head(105, 0, 5); key_head(124, 0, 0)
bpy.ops.object.mode_set(mode='OBJECT')

# ease in / out on every curve
for ad in (body.data.shape_keys.animation_data, arm.animation_data):
    for fc_ in ad.action.fcurves:
        for kp in fc_.keyframe_points:
            kp.interpolation = 'BEZIER'
            kp.easing = 'EASE_IN_OUT'
            kp.handle_left_type = kp.handle_right_type = 'AUTO_CLAMPED'

bpy.ops.wm.save_as_mainfile(filepath="/tmp/lp_driver_test/mblab_anim.blend")

if mode == "still":
    f = int(argv[1]) if len(argv) > 1 else 68
    scn.frame_set(f)
    scn.render.filepath = f"/tmp/lp_driver_test/still_{f:04d}.png"
    bpy.ops.render.render(write_still=True)
    print("STILL", scn.render.filepath)
else:
    scn.render.filepath = "/tmp/lp_driver_test/frames/frame_"
    bpy.ops.render.render(animation=True)
    print("ANIM done")
