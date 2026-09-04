"""Shared scene helpers for the driving-clip pipeline: load a finalized
MB-Lab character, repath its textures, measure the head, and build the
camera / lights / backdrop / render settings every clip is rendered with.

Import from a script run as ``Blender --background --python x.py`` after
adding this folder to ``sys.path`` (see ``mblab_rich.py``).
"""
import os, math
import bpy
from mathutils import Vector, Euler

BODY_NAME = "LP_body"
ARM_NAME = "LP_armature"

# The MB-Lab data folder of the installed addon (textures live here).
MBLAB_TEX = os.path.expanduser(
    "~/Library/Application Support/Blender/4.5/scripts/addons/MB-Lab/data/textures")

# Framing family shared by every clip: 85 mm, chin->crown = 85 % of a
# 512x512 frame (face ~60 %), neutral grey backdrop, soft three-point.
RES = 512
FPS = 25
FOCAL_MM = 85.0
SENSOR_MM = 36.0
HEAD_FRACTION = 0.85
BG_WORLD = (0.32, 0.32, 0.33, 1.0)
BG_WORLD_STRENGTH = 0.6
BG_PLANE = (0.36, 0.36, 0.37, 1.0)


def open_character(path):
    bpy.ops.wm.open_mainfile(filepath=path)
    repath_images()
    return bpy.context.scene, bpy.data.objects[BODY_NAME], bpy.data.objects[ARM_NAME]


def repath_images():
    """humanoid_library.blend ships absolute iris texture paths into a 4.1
    addons dir; point every missing FILE image at the installed copy."""
    for img in bpy.data.images:
        if not img.has_data and img.source == 'FILE':
            cand = os.path.join(MBLAB_TEX, os.path.basename(img.filepath))
            if os.path.isfile(cand):
                img.filepath = cand
                img.reload()


class HeadMeasure:
    """Rest-pose landmarks of the character in world space (armature at the
    origin, character facing -Y, +Z up)."""

    def __init__(self, body, arm):
        self.body, self.arm = body, arm
        mw = body.matrix_world
        bones = arm.data.bones
        self.neck_z = (arm.matrix_world @ bones["neck"].head_local).z
        self.head_top = (arm.matrix_world @ bones["head"].tail_local).z
        verts = [mw @ v.co for v in body.data.vertices]
        face_verts = [v for v in verts if v.z > self.neck_z + 0.02]
        xs = [v.x for v in face_verts]; ys = [v.y for v in face_verts]; zs = [v.z for v in face_verts]
        self.minv = Vector((min(xs), min(ys), min(zs)))
        self.maxv = Vector((max(xs), max(ys), max(zs)))
        self.head_h = self.maxv.z - self.minv.z
        self.face_front_y = self.minv.y
        self.face_center = Vector(((self.minv.x + self.maxv.x) / 2, self.minv.y,
                                   self.minv.z + self.head_h * 0.47))
        # eyes: vertices of the sclera material slot, split by side
        me = body.data
        eye_slot = next(i for i, m in enumerate(me.materials) if m.name.endswith("human_eyes"))
        eye_vs = set()
        for p in me.polygons:
            if p.material_index == eye_slot:
                eye_vs.update(p.vertices)
        eye_pts = [mw @ me.vertices[i].co for i in eye_vs]
        left = [p for p in eye_pts if p.x > 0]     # character's left is +X (she faces -Y)
        right = [p for p in eye_pts if p.x < 0]
        self.eye_l = sum(left, Vector()) / len(left)
        self.eye_r = sum(right, Vector()) / len(right)
        self.eye_z = (self.eye_l.z + self.eye_r.z) / 2
        self.eye_r_radius = max((p - self.eye_r).length for p in right)
        # skull centre estimate: middle of the head box in x/y, a little above the eyes
        self.skull_center = Vector(((self.minv.x + self.maxv.x) / 2,
                                    (self.minv.y + self.maxv.y) / 2 + 0.005,
                                    self.eye_z + 0.04))

    def describe(self):
        return ("head box %s..%s h=%.3f eye_z=%.3f neck_z=%.3f eyeL=%s eyeR=%s skull=%s" % (
            tuple(round(c, 3) for c in self.minv), tuple(round(c, 3) for c in self.maxv), self.head_h,
            self.eye_z, self.neck_z, tuple(round(c, 3) for c in self.eye_l),
            tuple(round(c, 3) for c in self.eye_r), tuple(round(c, 3) for c in self.skull_center)))


def setup_camera(scn, hm):
    """Fixed framing for every clip. Returns the camera object."""
    frame_h = hm.head_h / HEAD_FRACTION
    dist = frame_h * FOCAL_MM / SENSOR_MM
    cam_data = bpy.data.cameras.new("LPCam")
    cam_data.lens = FOCAL_MM
    cam_data.sensor_width = SENSOR_MM
    cam_data.sensor_height = SENSOR_MM
    cam_data.sensor_fit = 'VERTICAL'
    cam_data.clip_start = 0.05
    cam = bpy.data.objects.new("LPCam", cam_data)
    scn.collection.objects.link(cam)
    fc = hm.face_center
    cam.location = Vector((fc.x, hm.face_front_y - dist, fc.z))
    cam.rotation_euler = Euler((math.radians(90), 0, 0), 'XYZ')  # look along +Y
    scn.camera = cam
    return cam, frame_h


def _area(scn, name, loc, target, energy, size, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, 'AREA')
    d.energy = energy; d.size = size; d.color = color
    o = bpy.data.objects.new(name, d)
    scn.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return o


def setup_lights(scn, hm):
    fc = hm.face_center
    _area(scn, "Key",  (fc.x + 0.9, fc.y - 1.1, fc.z + 0.7), fc, 60, 1.0, (1.0, 0.96, 0.92))
    _area(scn, "Fill", (fc.x - 1.1, fc.y - 1.0, fc.z + 0.2), fc, 25, 1.5, (0.92, 0.95, 1.0))
    _area(scn, "Rim",  (fc.x - 0.4, fc.y + 0.9, fc.z + 0.8), fc, 30, 0.6)
    # a small frontal catch light so the eyes carry a wet highlight
    _area(scn, "Catch", (fc.x + 0.15, fc.y - 1.6, fc.z + 0.25), fc, 6, 0.25, (1.0, 0.98, 0.95))


def setup_backdrop(scn, hm):
    fc = hm.face_center
    world = bpy.data.worlds.new("LPWorld"); scn.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = BG_WORLD
    bg.inputs[1].default_value = BG_WORLD_STRENGTH
    me = bpy.data.meshes.new("Backdrop")
    s = 3.0
    me.from_pydata([(-s, 0, -s), (s, 0, -s), (s, 0, s), (-s, 0, s)], [], [(0, 1, 2, 3)])
    plane = bpy.data.objects.new("Backdrop", me)
    scn.collection.objects.link(plane)
    plane.location = (fc.x, fc.y + 2.5, fc.z)
    pm = bpy.data.materials.new("BackdropGrey"); pm.use_nodes = True
    bsdf = pm.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = BG_PLANE
    bsdf.inputs["Roughness"].default_value = 1.0
    me.materials.append(pm)
    return plane


def setup_render(scn, frame_start=0, frame_end=0, samples=48):
    scn.render.engine = 'BLENDER_EEVEE_NEXT'
    scn.render.resolution_x = RES
    scn.render.resolution_y = RES
    scn.render.resolution_percentage = 100
    scn.render.fps = FPS
    scn.frame_start = frame_start
    scn.frame_end = frame_end
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGB'
    scn.render.film_transparent = False
    scn.eevee.taa_render_samples = samples
    scn.render.filter_size = 1.8      # softer pixel filter: thin strands alias less between frames
    scn.eevee.use_shadows = True
    scn.eevee.use_raytracing = False
    # hair as strands, a few subdivisions so curved strands stay smooth
    scn.render.hair_type = 'STRAND'
    scn.render.hair_subdiv = 1
    scn.view_settings.view_transform = 'AgX'
    scn.view_settings.look = 'None'


def render_frames(scn, out_dir, frames):
    """Render an explicit list of frame numbers as PNG stills into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    for f in frames:
        scn.frame_set(f)
        scn.render.filepath = os.path.join(out_dir, f"frame_{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    return [os.path.join(out_dir, f"frame_{f:04d}.png") for f in frames]
