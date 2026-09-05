"""Dump what a finalized MB-Lab character blend contains: objects, bones,
shape keys, materials and image paths.  Run headless:

  Blender --background --python inspect_char.py -- /path/to/char.blend
"""
import bpy, sys

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
path = argv[0] if argv else "mblab_char.blend"
bpy.ops.wm.open_mainfile(filepath=path)

for o in bpy.data.objects:
    print("OBJ", o.name, o.type, "parent=", o.parent.name if o.parent else None,
          "dims=", tuple(round(d, 3) for d in o.dimensions), "loc=", tuple(round(c, 3) for c in o.location))
    if o.type == 'MESH':
        me = o.data
        print("   verts", len(me.vertices), "polys", len(me.polygons), "mats", [m.name for m in me.materials])
        print("   vgroups", len(o.vertex_groups), "uv", [u.name for u in me.uv_layers])
        print("   modifiers", [(m.name, m.type) for m in o.modifiers])
        if me.shape_keys:
            names = [k.name for k in me.shape_keys.key_blocks]
            print("   shapekeys", len(names))
            for n in names:
                print("      SK", n)
    if o.type == 'ARMATURE':
        for b in o.data.bones:
            print("   BONE", b.name, "parent=", b.parent.name if b.parent else None,
                  "head=", tuple(round(c, 3) for c in b.head_local), "tail=", tuple(round(c, 3) for c in b.tail_local))

for m in bpy.data.materials:
    if not m.use_nodes:
        print("MAT", m.name, "(no nodes)"); continue
    print("MAT", m.name)
    for n in m.node_tree.nodes:
        extra = ""
        if n.type == 'TEX_IMAGE' and n.image:
            extra = f" img={n.image.name} path={n.image.filepath} has_data={n.image.has_data}"
        if n.type == 'BSDF_PRINCIPLED':
            vals = {}
            for k in ("Subsurface Weight", "Subsurface Radius", "Subsurface Scale", "Roughness", "Specular IOR Level", "Coat Weight"):
                if k in n.inputs:
                    v = n.inputs[k].default_value
                    vals[k] = tuple(round(x, 3) for x in v) if hasattr(v, "__len__") else round(v, 3)
            extra = " " + str(vals)
        print("   NODE", n.name, n.type, extra)
    for l in m.node_tree.links:
        print("   LINK", l.from_node.name, ".", l.from_socket.name, "->", l.to_node.name, ".", l.to_socket.name)

for img in bpy.data.images:
    print("IMG", img.name, img.filepath, img.has_data, tuple(img.size))

# material slots -> polygon counts (which faces use which material)
body = next((o for o in bpy.data.objects if o.type == 'MESH' and o.data.shape_keys), None)
if body:
    counts = {}
    for p in body.data.polygons:
        counts[p.material_index] = counts.get(p.material_index, 0) + 1
    for i, c in sorted(counts.items()):
        print("SLOT", i, body.data.materials[i].name if i < len(body.data.materials) else None, "polys", c)
    for vg in body.vertex_groups:
        print("VG", vg.name)
