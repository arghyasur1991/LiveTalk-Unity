"""Numeric judge: consecutive-frame MAD (jitter proxy), first-vs-last frame MAD,
and a contact sheet of neutral / mid-yaw / blink / peak-smile frames."""
import sys, os, glob
import numpy as np
from PIL import Image

def load(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    return files, [np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) for f in files]

def report(name, folder, picks):
    files, imgs = load(folder)
    if len(imgs) < 2:
        print(f"{name}: only {len(imgs)} frames"); return
    diffs = np.array([np.abs(imgs[i+1] - imgs[i]).mean() for i in range(len(imgs)-1)])
    first_last = np.abs(imgs[-1] - imgs[0]).mean()
    # face-region-only variant (centre 60% box) to down-weight static background
    h, w = imgs[0].shape[:2]
    y0, y1, x0, x1 = int(h*0.15), int(h*0.85), int(w*0.2), int(w*0.8)
    cdiffs = np.array([np.abs(imgs[i+1][y0:y1, x0:x1] - imgs[i][y0:y1, x0:x1]).mean() for i in range(len(imgs)-1)])
    print(f"{name}: n={len(imgs)} size={w}x{h}")
    print(f"  consecutive MAD (0-255): median={np.median(diffs):.3f} mean={diffs.mean():.3f} p90={np.percentile(diffs,90):.3f} max={diffs.max():.3f} (argmax frame {diffs.argmax()}->{diffs.argmax()+1})")
    print(f"  centre-crop MAD:          median={np.median(cdiffs):.3f} max={cdiffs.max():.3f}")
    print(f"  frame0 vs frame{len(imgs)-1} MAD: {first_last:.3f}")
    # spikes: frames whose diff is > 3x median (oscillation / pop detector)
    spikes = np.where(diffs > 3*np.median(diffs))[0]
    print(f"  spikes (>3x median): {len(spikes)} at {spikes.tolist()[:20]}")
    sel = [imgs[min(p, len(imgs)-1)] for p in picks]
    sheet = np.concatenate(sel, axis=1).astype(np.uint8)
    out = os.path.join(os.path.dirname(folder.rstrip('/')), f"contact_{name}.png")
    Image.fromarray(sheet).save(out)
    print(f"  contact sheet -> {out} (frames {picks})")
    np.savetxt(os.path.join(os.path.dirname(folder.rstrip('/')), f"diffs_{name}.csv"), diffs, fmt="%.4f")

if __name__ == "__main__":
    picks = [0, 25, 43, 68]
    for name, folder in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
        report(name, folder, picks)
