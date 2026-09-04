"""Turn the bald finalized MB-Lab character into a person: authored
eyebrows, eyelashes and a short swept-back hairstyle as hair Curves that
ride the skin through *Deform Curves on Surface*, subsurface skin, a wet
cornea, and a crew-neck top.  Also installs the shared camera / lights /
backdrop.

  Blender --background --python mblab_rich.py -- \
      --src work/mblab_char.blend --out work/mblab_char_rich.blend \
      [--test work/rich_test] [--flicker 10]

--test renders five poses (neutral, open mouth + brows, smile, yaw, blink +
pitch) so the look can be judged before any clip is rendered.  --flicker N
renders N consecutive frames of a slow yaw for a frame-to-frame stability
check of the hair.

Why Curves and not a particle system: particle hair keys can be written
(`ParticleHairKey.co_object_set`) but in background mode the original
particle system never receives them, so a combed style is lost on save.
Hair Curves store their points in the datablock.
"""
import os, sys, math, random, argparse
import bpy
from mathutils import Vector, noise

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_scene as S  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--test", default=None, help="folder for test stills")
ap.add_argument("--flicker", type=int, default=0)
ap.add_argument("--samples", type=int, default=48)
args = ap.parse_args(argv)

# ---------------------------------------------------------------- look knobs
HAIR_COLOR = (0.052, 0.030, 0.017, 1.0)      # dark brown
HAIR_STRANDS = 11000
HAIR_POINTS = 8
HAIR_ROOT_RADIUS = 0.00048                  # fat enough to cover a pixel: thinner strands shimmer frame to frame
HAIR_TIP_RADIUS = 0.00018
BROW_STRANDS = 440                           # per brow
BROW_ROOT_RADIUS = 0.00017
LASH_UPPER_PER_ROOT = 1
LASH_LOWER_PER_ROOT = 0
LASH_RADIUS = 0.00011
TOP_COLOR = (0.075, 0.105, 0.150, 1.0)       # slate blue cotton
SKIN_SSS_WEIGHT = 0.22
SKIN_SSS_SCALE = 0.006
SKIN_ROUGHNESS = 0.46

scn, body, arm = S.open_character(args.src)
hm = S.HeadMeasure(body, arm)
print("[rich]", hm.describe())
me = body.data
rng = random.Random(7)
C = hm.skull_center


def slot_index(suffix):
    return next(i for i, m in enumerate(me.materials) if m.name.endswith(suffix))


def verts_of_slot(idx):
    vs = set()
    for p in me.polygons:
        if p.material_index == idx:
            vs.update(p.vertices)
    return vs


# ---------------------------------------------------------------- materials
def hair_material(name, color, roughness=0.32, tip_lighten=0.25, random_dark=0.55):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.25
    bsdf.inputs["Coat Weight"].default_value = 0.0
    hi = nt.nodes.new("ShaderNodeHairInfo")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = tuple(c * random_dark for c in color[:3]) + (1,)
    ramp.color_ramp.elements[1].color = color
    nt.links.new(hi.outputs["Random"], ramp.inputs["Fac"])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = 'RGBA'
    mix.inputs[7].default_value = tuple(min(1, c * 1.9) for c in color[:3]) + (1,)
    nt.links.new(ramp.outputs["Color"], mix.inputs[6])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = tip_lighten
    nt.links.new(hi.outputs["Intercept"], mul.inputs[0])
    nt.links.new(mul.outputs[0], mix.inputs["Factor"])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return m


def tune_skin():
    g = bpy.data.node_groups["MBLab_Skin3"]
    bsdf = next(n for n in g.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs["Subsurface Weight"].default_value = SKIN_SSS_WEIGHT
    bsdf.inputs["Subsurface Scale"].default_value = SKIN_SSS_SCALE
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.18)
    bsdf.inputs["Roughness"].default_value = SKIN_ROUGHNESS
    bsdf.subsurface_method = 'RANDOM_WALK'
    # The material's Displacement output (fed by the 2048 displacement map)
    # is rendered as bump by EEVEE and produces blocky red patches at the
    # nostrils and eye corners; the Displace *modifier* already applies
    # that map to the geometry, so drop the shader-side copy.
    nt = bpy.data.materials["LP_MBLab_skin3"].node_tree
    for l in list(nt.links):
        if l.to_socket.name == "Displacement":
            nt.links.remove(l)


def tune_eyes():
    cornea = bpy.data.materials["LP_MBlab_cornea"]
    b = next(n for n in cornea.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs["Specular IOR Level"].default_value = 0.6
    b.inputs["IOR"].default_value = 1.376
    b.inputs["Roughness"].default_value = 0.02
    b.inputs["Coat Weight"].default_value = 0.6
    b.inputs["Coat Roughness"].default_value = 0.02
    cornea.surface_render_method = 'BLENDED'
    cornea.use_transparency_overlap = True
    eyes = bpy.data.materials["LP_MBlab_human_eyes"]
    for n in eyes.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            n.inputs["Roughness"].default_value = 0.08
            n.inputs["Specular IOR Level"].default_value = 0.55
    lash = bpy.data.materials["LP_MBlab_eyelash"]
    lash.surface_render_method = 'BLENDED'   # hashed alpha leaves pink speckle under the eyes
    for n in lash.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            n.inputs["Roughness"].default_value = 0.4


def tune_teeth():
    for name in ("LP_MBlab_human_teeth", "LP_MBLab_tongue"):
        for n in bpy.data.materials[name].node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                n.inputs["Subsurface Weight"].default_value = 0.15
                n.inputs["Subsurface Scale"].default_value = 0.004


# ---------------------------------------------------------------- surface sampling
uv_layer = me.uv_layers["UVMap"].data


def ensure_rest_position():
    if "rest_position" not in me.attributes:
        a = me.attributes.new("rest_position", 'FLOAT_VECTOR', 'POINT')
        a.data.foreach_set("vector", [c for v in me.vertices for c in v.co])


def face_sample(p, w_tri):
    """Random point + normal + UV on polygon p (fan triangulation)."""
    li = list(p.loop_indices)
    vi = [me.loops[l].vertex_index for l in li]
    k = rng.randrange(1, len(vi) - 1)
    ia, ib, ic = 0, k, k + 1
    r1, r2 = rng.random(), rng.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    w = (1 - r1 - r2, r1, r2)
    pos = me.vertices[vi[ia]].co * w[0] + me.vertices[vi[ib]].co * w[1] + me.vertices[vi[ic]].co * w[2]
    uv = uv_layer[li[ia]].uv * w[0] + uv_layer[li[ib]].uv * w[1] + uv_layer[li[ic]].uv * w[2]
    return pos, p.normal.copy(), uv


def uv_at(poly_index, pos):
    """UV of a point on polygon poly_index (barycentric over the fan)."""
    p = me.polygons[poly_index]
    li = list(p.loop_indices)
    vi = [me.loops[l].vertex_index for l in li]
    best = None
    for k in range(1, len(vi) - 1):
        a, b, c = me.vertices[vi[0]].co, me.vertices[vi[k]].co, me.vertices[vi[k + 1]].co
        v0, v1, v2 = b - a, c - a, pos - a
        d00, d01, d11, d20, d21 = v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1)
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-12:
            continue
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        u = 1 - v - w
        err = max(0, -u) + max(0, -v) + max(0, -w)
        if best is None or err < best[0]:
            best = (err, uv_layer[li[0]].uv * u + uv_layer[li[k]].uv * v + uv_layer[li[k + 1]].uv * w)
    return best[1]


_bvh = None


def skin_hit(origin, direction):
    """Ray cast against the *base* body mesh (Object.ray_cast would hit the
    subdivided evaluated mesh and return its face indices); returns
    (pos, normal, uv) or None."""
    global _bvh
    if _bvh is None:
        from mathutils.bvhtree import BVHTree
        _bvh = BVHTree.FromPolygons([v.co for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
    loc, nrm, idx, dist = _bvh.ray_cast(origin, direction)
    if loc is None:
        return None
    return loc, nrm, uv_at(idx, loc)


def scalp_polys():
    """Hairline: forehead ~6.3 cm above the eyes (higher at the temples),
    above the ears on the sides, down to the nape at the back."""
    skin = slot_index("skin3")
    out = []
    for p in me.polygons:
        if p.material_index != skin:
            continue
        c = p.center
        if c.z < hm.neck_z + 0.015:
            continue
        dz = c.z - hm.eye_z
        dx, dy = c.x - C.x, c.y - C.y
        phi = math.atan2(abs(dx), -dy)
        if phi < math.radians(55):
            temple = max(0.0, (abs(dx) - 0.04) / 0.03)
            ok = dz > 0.063 + 0.012 * temple
        elif phi < math.radians(115):
            ok = dz > 0.036 and not (abs(dx) > 0.066 and dz < 0.055)   # skip ears
        else:
            ok = dz > -0.05
        if ok:
            out.append(p)
    return out


# ---------------------------------------------------------------- curves builder
def make_curves(name, strands, material, parent=body):
    """strands: list of (points[list[Vector]], uv, radii[list[float]])."""
    cd = bpy.data.hair_curves.new(name)
    cd.add_curves([len(pts) for pts, _, _ in strands])
    co, rad, uvs = [], [], []
    for pts, uv, radii in strands:
        for p in pts:
            co.extend(p)
        rad.extend(radii)
        uvs.extend((uv.x, uv.y))
    cd.attributes["position"].data.foreach_set("vector", co)
    cd.attributes.new("radius", 'FLOAT', 'POINT').data.foreach_set("value", rad)
    cd.attributes.new("surface_uv_coordinate", 'FLOAT2', 'CURVE').data.foreach_set("vector", uvs)
    cd.surface = body
    cd.surface_uv_map = "UVMap"
    cd.materials.append(material)
    ob = bpy.data.objects.new(name, cd)
    scn.collection.objects.link(ob)
    ob.parent = parent
    ng = bpy.data.node_groups.get("LP_DeformOnSurface")
    if ng is None:
        ng = bpy.data.node_groups.new("LP_DeformOnSurface", 'GeometryNodeTree')
        ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        gi = ng.nodes.new("NodeGroupInput"); go = ng.nodes.new("NodeGroupOutput")
        d = ng.nodes.new("GeometryNodeDeformCurvesOnSurface")
        ng.links.new(gi.outputs[0], d.inputs[0]); ng.links.new(d.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("Deform", 'NODES'); mod.node_group = ng
    print("[rich] curves", name, len(strands), "strands", len(co) // 3, "points")
    return ob


def radii(n, root, tip):
    return [root + (tip - root) * (i / (n - 1)) ** 0.8 for i in range(n)]


def coherent(v, scale, seed_off):
    """Spatially coherent random vector in [-1,1]^3 (neighbouring roots get
    similar jitter -> natural clumping without hard structure)."""
    p = v * scale + Vector((seed_off, seed_off * 1.7, seed_off * 0.3))
    return Vector((noise.noise(p), noise.noise(p + Vector((31.7, 0, 0))), noise.noise(p + Vector((0, 47.1, 0)))))


DOWN = Vector((0, 0, -1))


def head_hair_strands():
    polys = scalp_polys()
    area = sum(p.area for p in polys)
    strands = []
    for p in polys:
        n_here = max(1, round(HAIR_STRANDS * p.area / area))
        for _ in range(n_here):
            r, n, uv = face_sample(p, None)
            rel = r - C
            dz = r.z - hm.eye_z
            phi = math.atan2(abs(rel.x), -rel.y)
            radial_xy = Vector((rel.x, rel.y, 0)).length
            jit = coherent(r, 40.0, 3.0) * 0.045 + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))) * 0.012
            front = phi < math.radians(58)
            crown = dz > 0.085
            if front:
                length = 0.06 + 0.02 * rng.random()
                target = Vector((0.35 * math.copysign(1, rel.x) + 0.2 * (abs(rel.x) / 0.06), 1.0, -0.2)).normalized()
                pad_end = 0.004
            elif crown:
                length = 0.09 + 0.025 * rng.random()
                target = (Vector((rel.x, rel.y, 0)).normalized() * 0.45 + DOWN * 0.9).normalized()
                pad_end = 0.005
            else:
                length = 0.11 + 0.03 * rng.random()
                target = (DOWN + Vector((rel.x, rel.y, 0)).normalized() * 0.08).normalized()
                pad_end = 0.004
            k = HAIR_POINTS
            seg = length / (k - 1)
            pos = r.copy(); pts = [pos.copy()]
            wave_phase = rng.random() * 6.283
            wave_amp = 0.0005 + 0.0008 * rng.random()
            side = Vector((-rel.y, rel.x, 0)).normalized() if radial_xy > 1e-4 else Vector((1, 0, 0))
            for i in range(1, k):
                t = i / (k - 1)
                w = t ** 0.75
                d = ((1 - w) * n + w * (target + jit)).normalized()
                pos = pos + d * seg
                pad = 0.002 + pad_end * w
                if front or pos.z > C.z + 0.015:
                    # spherical clamp around the skull centre over the top
                    rr = pos - C
                    rmin = rel.length + pad
                    if rr.length < rmin:
                        pos = C + rr.normalized() * rmin
                else:
                    # cylindrical clamp down the sides / back
                    rr = Vector((pos.x - C.x, pos.y - C.y, 0))
                    rmin = radial_xy + pad
                    if rr.length < rmin:
                        pos = Vector((C.x, C.y, pos.z)) + rr.normalized() * rmin
                pts.append(pos + side * (math.sin(wave_phase + t * 5.0) * wave_amp * t))
            strands.append((pts, uv, radii(k, HAIR_ROOT_RADIUS, HAIR_TIP_RADIUS)))
    return strands


def brow_strands(sign):
    """One eyebrow as an arc of skin-hugging hairs. sign=+1 character's
    left (+X), -1 right. Landmarks are relative to the eye centre."""
    eye = hm.eye_l if sign > 0 else hm.eye_r
    strands = []
    n_pts = 5
    for _ in range(BROW_STRANDS):
        s = rng.random() ** 0.9                    # denser toward the inner end
        x = sign * (0.014 + 0.040 * s)
        arch = math.sin(math.pi * min(1.0, s / 0.92) ** 0.85)
        zc = hm.eye_z + 0.0165 + 0.0062 * arch
        half_h = 0.0026 * (1 - 0.6 * s) + 0.0006
        z = zc + rng.uniform(-half_h, half_h)
        hit = skin_hit(Vector((x, hm.face_front_y - 0.08, z)), Vector((0, 1, 0)))
        if hit is None:
            continue
        r, n, uv = hit
        # tangent along the arc toward the tail; inner hairs stand up more
        dzds = 0.0085 * (math.cos(math.pi * s) * math.pi) * 0.043 / 0.043
        tangent = Vector((sign * 1.0, 0, dzds * 0.9)).normalized()
        up = 0.8 * (1 - s) ** 1.6 - 0.15 * s
        d0 = (tangent + Vector((0, 0, up)) + n * 0.3).normalized()
        d1 = (tangent + Vector((0, 0, up * 0.3 - 0.2 * s)) + n * 0.0).normalized()
        length = 0.0065 - 0.002 * s + rng.uniform(-0.0006, 0.0006)
        pts = [r.copy()]; pos = r.copy()
        for i in range(1, n_pts):
            t = i / (n_pts - 1)
            d = ((1 - t) * d0 + t * d1).normalized()
            pos = pos + d * (length / (n_pts - 1))
            pts.append(pos.copy())
        strands.append((pts, uv, radii(n_pts, BROW_ROOT_RADIUS, BROW_ROOT_RADIUS * 0.35)))
    return strands


def lash_strands():
    """Roots at the eyelash-card vertices that sit on the lid margin (about
    one eye radius from the eye centre); upper lashes sweep forward and
    up, lower ones forward and down."""
    lash_vs = verts_of_slot(slot_index("eyelash"))
    # uv per vertex (first loop)
    vuv = {}
    for p in me.polygons:
        for li in p.loop_indices:
            vuv.setdefault(me.loops[li].vertex_index, uv_layer[li].uv)
    strands = []
    n_pts = 5
    for i in lash_vs:
        r = me.vertices[i].co.copy()
        eye = hm.eye_l if r.x > 0 else hm.eye_r
        dist = (r - eye).length
        if abs(dist - hm.eye_r_radius) > 0.0022:
            continue                              # card outer edge, not the lid margin
        upper = r.z > eye.z + 0.001
        radial = (r - eye).normalized()
        count = LASH_UPPER_PER_ROOT if upper else LASH_LOWER_PER_ROOT
        for _ in range(count):
            lateral = math.copysign(1, r.x - eye.x) * 0.25 * abs(r.x - eye.x) / max(1e-4, hm.eye_r_radius)
            if upper:
                d0 = (radial * 0.6 + Vector((lateral, -0.6, 0.35))).normalized()
                d1 = (radial * 0.1 + Vector((lateral, -0.3, 1.0))).normalized()
                length = 0.0055 + rng.uniform(-0.0007, 0.0008)
            else:
                d0 = (radial * 0.6 + Vector((lateral, -0.6, -0.4))).normalized()
                d1 = (radial * 0.1 + Vector((lateral, -0.3, -1.0))).normalized()
                length = 0.0038 + rng.uniform(-0.0005, 0.0006)
            jitter = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))) * 0.12
            pts = [r.copy()]; pos = r.copy()
            for k in range(1, n_pts):
                t = k / (n_pts - 1)
                d = ((1 - t) * d0 + t * d1 + jitter).normalized()
                pos = pos + d * (length / (n_pts - 1))
                pts.append(pos.copy())
            strands.append((pts, vuv[i], radii(n_pts, LASH_RADIUS, LASH_RADIUS * 0.3)))
    return strands


# ---------------------------------------------------------------- garment
def add_top():
    """Crew-neck top: a copy of the torso/shoulder skin pushed 4.5 mm out
    with a fabric material. Keeps the armature modifier + shape keys so it
    breathes with the chest."""
    import bmesh
    top = body.copy()
    top.data = me.copy()
    top.name = "LP_top"; top.data.name = "LP_top"
    scn.collection.objects.link(top)
    top.parent = arm
    top.matrix_parent_inverse = body.matrix_parent_inverse.copy()
    tme = top.data
    keep = set()
    for v in tme.vertices:
        p = v.co
        frontness = max(0.0, min(1.0, -(p.y - C.y) / 0.08))
        # crew neck: high at the back, dipping ~4 cm at the front
        neckline = hm.neck_z + 0.048 - 0.042 * frontness
        neck_col = Vector((p.x - C.x, p.y - C.y, 0)).length < 0.064 and p.z > hm.neck_z - 0.004
        if p.z < neckline and not neck_col:
            keep.add(v.index)
    bm = bmesh.new(); bm.from_mesh(tme)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bv for bv in bm.verts if bv.index not in keep], context='VERTS')
    bm.to_mesh(tme); bm.free()
    for m in list(top.modifiers):
        if m.type in ('PARTICLE_SYSTEM', 'DISPLACE', 'CORRECTIVE_SMOOTH'):
            top.modifiers.remove(m)
    disp = top.modifiers.new("Offset", 'DISPLACE')
    # the body's own displacement map moves skin +-5 mm, so the cloth must
    # sit further out than that or the skin pokes through in blobs
    disp.strength = 0.011; disp.mid_level = 0.0; disp.direction = 'NORMAL'
    sol = top.modifiers.new("Thickness", 'SOLIDIFY')
    sol.thickness = 0.0025; sol.offset = -1.0
    tme.materials.clear()
    fab = bpy.data.materials.new("LP_fabric"); fab.use_nodes = True
    nt = fab.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = TOP_COLOR
    b.inputs["Roughness"].default_value = 0.88
    b.inputs["Specular IOR Level"].default_value = 0.25
    b.inputs["Sheen Weight"].default_value = 0.35
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 900.0; nz.inputs["Detail"].default_value = 2.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35; bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    tme.materials.append(fab)
    for p in tme.polygons:
        p.material_index = 0
    print("[rich] top: kept", len(keep), "verts")
    return top


# ================================================================ build
tune_skin(); tune_eyes(); tune_teeth()
ensure_rest_position()

brow_mat = hair_material("LP_hair_brow", (0.055, 0.033, 0.019, 1.0), roughness=0.45, tip_lighten=0.2)
lash_mat = hair_material("LP_hair_lash", (0.016, 0.010, 0.008, 1.0), roughness=0.45, tip_lighten=0.0, random_dark=0.8)
head_mat = hair_material("LP_hair_head", HAIR_COLOR, roughness=0.42, tip_lighten=0.15)

make_curves("LP_browL", brow_strands(+1), brow_mat)
make_curves("LP_browR", brow_strands(-1), brow_mat)
make_curves("LP_lashes", lash_strands(), lash_mat)
make_curves("LP_hair", head_hair_strands(), head_mat)
add_top()

cam, frame_h = S.setup_camera(scn, hm)
S.setup_lights(scn, hm)
S.setup_backdrop(scn, hm)
S.setup_render(scn, 0, 0, samples=args.samples)
print("[rich] camera", tuple(round(c, 3) for c in cam.location), "frame_h %.3f" % frame_h)

for k in me.shape_keys.key_blocks:
    k.value = 0.0

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
print("[rich] saved", args.out)

# ---------------------------------------------------------------- test stills
if args.test:
    kb = me.shape_keys.key_blocks
    pb = arm.pose.bones["head"]
    pb.rotation_mode = 'XYZ'

    def pose(sk=None, head=(0, 0, 0)):
        for k in kb:
            k.value = 0.0
        for n, v in (sk or {}).items():
            kb[n].value = v
        pb.rotation_euler = tuple(math.radians(a) for a in head)

    poses = [
        ("neutral", {}, (0, 0, 0)),
        ("open", {"Expressions_mouthOpen_max": 0.55, "Expressions_browOutVertL_max": 1.0,
                  "Expressions_browOutVertR_max": 1.0, "Expressions_browsMidVert_max": 0.8}, (0, 0, 0)),
        ("smile", {"Expressions_mouthSmile_max": 1.0, "Expressions_cheekSneerL_max": 0.1,
                   "Expressions_cheekSneerR_max": 0.1, "Expressions_eyeSquintL_max": 0.4,
                   "Expressions_eyeSquintR_max": 0.4}, (0, 0, 0)),
        ("yaw", {}, (0, -10, 0)),
        ("blink_pitch", {"Expressions_eyeClosedL_max": 1.0, "Expressions_eyeClosedR_max": 1.0}, (8, 0, 4)),
    ]
    os.makedirs(args.test, exist_ok=True)
    import time
    for i, (name, sk, head) in enumerate(poses):
        pose(sk, head)
        scn.frame_set(i)
        t0 = time.time()
        scn.render.filepath = os.path.join(args.test, f"test_{i}_{name}.png")
        bpy.ops.render.render(write_still=True)
        print("[rich] still %s  %.1fs" % (scn.render.filepath, time.time() - t0))
    pose()
    if args.flicker:
        pb.rotation_euler = (0, 0, 0); pb.keyframe_insert("rotation_euler", frame=0)
        pb.rotation_euler = (0, math.radians(-4), 0); pb.keyframe_insert("rotation_euler", frame=args.flicker - 1)
        scn.frame_end = args.flicker - 1
        scn.render.filepath = os.path.join(args.test, "flicker", "frame_")
        bpy.ops.render.render(animation=True)
        print("[rich] flicker frames in", os.path.join(args.test, "flicker"))
