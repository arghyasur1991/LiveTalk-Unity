import bpy
bpy.ops.wm.open_mainfile(filepath="/tmp/lp_driver_test/mblab_char.blend")
for img in bpy.data.images:
    print("IMG", img.name, img.filepath, img.source, "size", tuple(img.size), "has_data", img.has_data, "packed", img.packed_file is not None)
body = bpy.data.objects["LP_body"]
for m in body.data.materials:
    if not m or not m.use_nodes: continue
    kinds = {}
    for n in m.node_tree.nodes:
        kinds[n.bl_idname] = kinds.get(n.bl_idname, 0) + 1
    print("MAT", m.name, kinds)
    for n in m.node_tree.nodes:
        if n.bl_idname == 'ShaderNodeTexImage':
            print("    tex", n.name, "->", n.image.name if n.image else None)
        if n.bl_idname == 'ShaderNodeEmission':
            print("    emission strength", n.inputs[1].default_value, "color", tuple(n.inputs[0].default_value))
        if n.bl_idname == 'ShaderNodeBsdfPrincipled':
            e = n.inputs.get("Emission Strength")
            print("    principled emission", e.default_value if e else None)
# eye material-slot usage: how many polys use each slot
counts = [0]*len(body.data.materials)
for p in body.data.polygons: counts[p.material_index] += 1
print("slot poly counts", list(zip([m.name for m in body.data.materials], counts)))
