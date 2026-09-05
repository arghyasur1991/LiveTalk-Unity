"""Numeric judge for a driving clip or for the LivePortrait output it drives.

  python analyze.py name=folder [name2=folder2 ...] [--picks 0,25,43,68] [--sheet-dir DIR]

Per folder of PNG frames (sorted by name) it prints:

* consecutive-frame mean absolute difference (0-255) — median / mean / p90 /
  max, plus a centre-crop variant that down-weights static background;
* first-vs-last frame MAD — an authored clip should give ~0 (same rest pose);
* the *peak* frame (largest face-region difference from frame 0) and the
  MAD of the outer border band at that frame — background must not move
  when the expression does (<= 0.1 is the bar). The top band catches hair
  when the head tilts, so the left/right side bands are reported too;
* spikes: consecutive differences above 3x the median (pops / oscillation).

It also writes a contact sheet ``contact_<name>.png`` (frames from --picks,
or six evenly spaced frames including the peak when --picks is omitted) and
``diffs_<name>.csv`` next to the folder (or into --sheet-dir).
"""
import sys, os, glob, argparse
import numpy as np
from PIL import Image


def load(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    return files, [np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) for f in files]


def report(name, folder, picks=None, sheet_dir=None):
    files, imgs = load(folder)
    if len(imgs) < 2:
        print(f"{name}: only {len(imgs)} frames"); return None
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    diffs = np.array([np.abs(imgs[i + 1] - imgs[i]).mean() for i in range(n - 1)])
    first_last = np.abs(imgs[-1] - imgs[0]).mean()
    y0, y1, x0, x1 = int(h * 0.15), int(h * 0.85), int(w * 0.2), int(w * 0.8)
    cdiffs = np.array([np.abs(imgs[i + 1][y0:y1, x0:x1] - imgs[i][y0:y1, x0:x1]).mean() for i in range(n - 1)])
    # peak = frame whose face region differs most from frame 0
    face0 = np.array([np.abs(im[y0:y1, x0:x1] - imgs[0][y0:y1, x0:x1]).mean() for im in imgs])
    peak = int(face0.argmax())
    band = int(min(h, w) * 0.08)
    mask = np.ones((h, w), dtype=bool); mask[band:h - band, band:w - band] = False
    bg_peak = np.abs(imgs[peak] - imgs[0])[mask].mean()
    bg_max = max(np.abs(im - imgs[0])[mask].mean() for im in imgs)
    # side bands only: the head never reaches the left/right edge, so this is
    # background proper (the top band catches hair when the head moves)
    side = np.zeros((h, w), dtype=bool); side[:, :band] = True; side[:, w - band:] = True
    side_peak = np.abs(imgs[peak] - imgs[0])[side].mean()
    side_max = max(np.abs(im - imgs[0])[side].mean() for im in imgs)
    spikes = np.where(diffs > 3 * np.median(diffs))[0]
    print(f"{name}: n={n} size={w}x{h}")
    print(f"  consecutive MAD (0-255): median={np.median(diffs):.3f} mean={diffs.mean():.3f} p90={np.percentile(diffs, 90):.3f} max={diffs.max():.3f} (argmax frame {diffs.argmax()}->{diffs.argmax() + 1})")
    print(f"  centre-crop MAD:          median={np.median(cdiffs):.3f} max={cdiffs.max():.3f}")
    print(f"  frame0 vs frame{n - 1} MAD: {first_last:.3f}")
    print(f"  peak frame {peak} (face diff vs 0 = {face0[peak]:.2f}); border-band MAD at peak = {bg_peak:.3f} (max {bg_max:.3f}); side-bands only = {side_peak:.3f} (max {side_max:.3f})")
    print(f"  spikes (>3x median): {len(spikes)} at {spikes.tolist()[:20]}")
    if picks is None:
        picks = sorted(set([0, n // 5, 2 * n // 5, peak, 3 * n // 5, 4 * n // 5, n - 1]))[:7]
    sel = [imgs[min(p, n - 1)] for p in picks]
    sheet = np.concatenate(sel, axis=1).astype(np.uint8)
    out_dir = sheet_dir or os.path.dirname(folder.rstrip('/'))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"contact_{name}.png")
    Image.fromarray(sheet).save(out)
    print(f"  contact sheet -> {out} (frames {picks})")
    np.savetxt(os.path.join(out_dir, f"diffs_{name}.csv"), diffs, fmt="%.4f")
    return dict(n=n, med=float(np.median(diffs)), max=float(diffs.max()), first_last=float(first_last),
                peak=peak, bg_peak=float(bg_peak), bg_max=float(bg_max), side_peak=float(side_peak), side_max=float(side_max), spikes=len(spikes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--picks", default=None)
    ap.add_argument("--sheet-dir", default=None)
    a = ap.parse_args()
    picks = [int(p) for p in a.picks.split(",")] if a.picks else None
    for pair in a.pairs:
        name, folder = pair.split("=", 1)
        report(name, folder, picks, a.sheet_dir)
