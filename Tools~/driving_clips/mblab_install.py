import bpy, addon_utils, shutil, os, sys, traceback

ADDONS = os.path.expanduser("~/Library/Application Support/Blender/4.5/scripts/addons")
SRC = "/tmp/mblab_src/MB-Lab-1_8_1"
DST = os.path.join(ADDONS, "MB-Lab")
if not os.path.isdir(DST):
    shutil.copytree(SRC, DST)
    print("copied MB-Lab to", DST)
else:
    print("MB-Lab already at", DST)

addon_utils.modules_refresh()
try:
    mod = addon_utils.enable("MB-Lab", default_set=False, persistent=False)
    print("enable ->", mod)
except Exception:
    traceback.print_exc()
    sys.exit(2)

print("has mbast.init_character:", hasattr(bpy.ops.mbast, "init_character"))
scn = bpy.context.scene
print("characters:", [i.identifier for i in scn.bl_rna.properties["mblab_character_name"].enum_items])
