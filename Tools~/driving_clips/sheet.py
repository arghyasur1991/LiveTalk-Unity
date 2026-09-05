"""Contact sheet of PNGs.

  python sheet.py OUT.png [--cols N] [--crop x0,y0,x1,y1] [--scale S] [--label] FILE_OR_GLOB...

Frames are laid out left-to-right, wrapping every --cols. --crop cuts the
same box from every frame before placing it (eyes/mouth close-ups),
--scale resizes the (cropped) tile, --label stamps the file stem.
"""
import sys, glob, os, argparse
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("inputs", nargs="+")
ap.add_argument("--cols", type=int, default=0)
ap.add_argument("--crop", default=None)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--label", action="store_true")
ap.add_argument("--pick", default=None, help="comma list of indices to keep (after sorting)")
a = ap.parse_args()

files = []
for pat in a.inputs:
    m = sorted(glob.glob(pat))
    files.extend(m if m else [pat])
if a.pick:
    idx = [int(i) for i in a.pick.split(",")]
    files = [files[min(i, len(files) - 1)] for i in idx]
tiles = []
for f in files:
    im = Image.open(f).convert("RGB")
    if a.crop:
        x0, y0, x1, y1 = [int(v) for v in a.crop.split(",")]
        im = im.crop((x0, y0, x1, y1))
    if a.scale != 1.0:
        im = im.resize((int(im.width * a.scale), int(im.height * a.scale)), Image.LANCZOS)
    if a.label:
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, im.width, 14), fill=(0, 0, 0))
        d.text((3, 1), os.path.splitext(os.path.basename(f))[0][:60], fill=(255, 255, 255))
    tiles.append(im)
cols = a.cols or len(tiles)
rows = (len(tiles) + cols - 1) // cols
tw, th = tiles[0].width, tiles[0].height
sheet = Image.new("RGB", (tw * cols, th * rows), (20, 20, 20))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
sheet.save(a.out)
print(a.out, f"{len(tiles)} tiles {sheet.width}x{sheet.height}")
