#!/usr/bin/env python3
"""Analytical validation of a driving clip against what LivePortrait rendered from it.

Both CSVs come from ``LiveTalkAPI.MeasureMotionAsync`` (one row per frame: pose,
scale, the 63-dim expression vector and the 203 tracked landmarks in that
image's own pixels).

Two questions, answered with numbers instead of eyeballing a contact sheet:

1. **Head size stability** of the rendered frames. Landmarks and feature
   matching both wobble 2-4 % around blinks and pitch, so neither can gate a
   size jump of that magnitude (both were tried). The gate is the motion
   extractor's own ``scale`` read off the *output*: the render is built from
   ``scale_source * (scale_d / scale_d0)``, so whatever range the extractor
   reports on the driver (its expression leakage floor: a jaw drop or brow
   raise reads as a 5-8 % bigger head with a fixed camera) is the most the
   output may show. Gate: output scale range (max/min) <= driver range *
   ``--scale-slack`` (default 1.15). The old relative-scale pipeline read 15 %
   on the output against 6 % on the driver; with ``ScaleTransfer = 0`` it
   reads 4.7 %. The outer-canthus span (``iod``) is printed for information.

2. **Expression / pose transfer**, on geometric features that mean the same
   thing on any face (all normalised by ``iod``): eye openness (LivePortrait's
   own ratios), lip openness, mouth width, mouth-corner drop (frown +, smile −),
   brow height per side split inner / outer (sad and confused move them in
   opposite directions), plus pitch / yaw / roll. Per feature: deltas from
   frame 0 in both sequences, Pearson correlation (moves *when* the driver
   moves) and amplitude ratio std(out)/std(driver) (moves *as much*). Rendered
   frames may be resampled / looped, so the driver is resampled to the output
   length. Features the driver holds still must stay still in the output.

Landmark layout is LivePortrait's 203-point ``landmark.onnx``: 0–23 left eye,
24–47 right eye (eye-close ratios use 6/18 over 0/12 and 30/42 over 24/36),
48/66 mouth corners, 90/102 upper/lower lip centre (lip-close ratio). Brow
points are found geometrically on frame 0 (above the eyes, within the eye span).

Exit code 0 = every gate passed, 1 = at least one failed.

    python compare_motion.py driver.csv output.csv [--scale-slack 1.15]
                              [--min-corr 0.8] [--amp-lo 0.6] [--amp-hi 1.6] [--list-steps]
"""
import argparse
import csv
import math
import sys


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"empty csv: {path}")
    keys = [k for k in rows[0].keys() if k != "frame"]
    return {k: [float(r[k]) for r in rows] for k in keys}, [r["frame"] for r in rows]


def landmarks(data, i):
    return [(data[f"lm{j}x"][i], data[f"lm{j}y"][i]) for j in range(203)]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mean_pt(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def brow_indices(lm):
    """Indices above the eyes within the eye span on this (frame-0) landmark set."""
    le, re = mean_pt(lm[0:24]), mean_pt(lm[24:48])
    iod = dist(le, re)
    eye_y = 0.5 * (le[1] + re[1])
    x_lo, x_hi = min(le[0], re[0]) - 0.6 * iod, max(le[0], re[0]) + 0.6 * iod
    mid_x = 0.5 * (le[0] + re[0])
    left, right = [], []
    for j in range(48, 203):
        x, y = lm[j]
        if x_lo <= x <= x_hi and 0.15 * iod <= eye_y - y <= 0.75 * iod:
            (left if x < mid_x else right).append(j)
    def split(idx):
        if not idx:
            return [], []
        xs = sorted(idx, key=lambda j: abs(lm[j][0] - mid_x))
        h = len(xs) // 2
        return xs[:h], xs[h:]          # inner (nearer the midline), outer
    return split(left), split(right)


def canthal_span(lm):
    """Outer-corner to outer-corner: the widest pair among the eye-width endpoints."""
    return max(dist(lm[a], lm[b]) for a in (0, 12) for b in (24, 36))


def features(data, brows):
    n = len(data["pitch"])
    out = {k: [] for k in ["iod", "eyeL", "eyeR", "lip", "mouthW", "corner",
                           "browLi", "browLo", "browRi", "browRo", "pitch", "yaw", "roll"]}
    for i in range(n):
        lm = landmarks(data, i)
        le, re = mean_pt(lm[0:24]), mean_pt(lm[24:48])
        iod = dist(le, re)
        eye_y = 0.5 * (le[1] + re[1])
        out["iod"].append(canthal_span(lm))
        out["eyeL"].append(dist(lm[6], lm[18]) / max(1e-6, dist(lm[0], lm[12])))
        out["eyeR"].append(dist(lm[30], lm[42]) / max(1e-6, dist(lm[24], lm[36])))
        mouth_w = dist(lm[48], lm[66])
        out["lip"].append(dist(lm[90], lm[102]) / max(1e-6, mouth_w))
        out["mouthW"].append(mouth_w / iod)
        out["corner"].append(((lm[48][1] + lm[66][1]) * 0.5 - (lm[90][1] + lm[102][1]) * 0.5) / iod)
        (bli, blo), (bri, bro) = brows
        for key, idx in (("browLi", bli), ("browLo", blo), ("browRi", bri), ("browRo", bro)):
            out[key].append((eye_y - mean_pt([lm[j] for j in idx])[1]) / iod if idx else 0.0)
        for k in ("pitch", "yaw", "roll"):
            out[k].append(data[k][i])
    return out


# Minimum driver std for a feature to count as "moving"; below it the output must also stay still.
MOVE_STD = {"eyeL": 0.03, "eyeR": 0.03, "lip": 0.02, "mouthW": 0.01, "corner": 0.006,
            "browLi": 0.008, "browLo": 0.008, "browRi": 0.008, "browRo": 0.008,
            "pitch": 0.5, "yaw": 0.5, "roll": 0.5}


def resample(series, n):
    m = len(series)
    if m == n:
        return list(series)
    out = []
    for i in range(n):
        t = i * (m - 1) / max(1, n - 1)
        j = min(int(math.floor(t)), m - 2)
        a = t - j
        out.append(series[j] * (1 - a) + series[j + 1] * a)
    return out


def std(xs):
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa, sb = std(a), std(b)
    if sa < 1e-9 or sb < 1e-9:
        return float("nan")
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n * sa * sb)


def scale_range_pct(data):
    return (max(data["scale"]) / min(data["scale"]) - 1) * 100


def size_report(name, feats, data, driver_range_pct=None, slack=1.15):
    h = feats["iod"]
    rng = scale_range_pct(data)
    verdict = ""
    ok = True
    if driver_range_pct is not None:
        ok = rng <= driver_range_pct * slack
        verdict = f"  (driver {driver_range_pct:.2f}%, allowed {driver_range_pct * slack:.2f}%) -> {'PASS' if ok else 'FAIL'}"
    print(f"[{name}] extractor scale {min(data['scale']):.4f}..{max(data['scale']):.4f} range {rng:.2f}%{verdict}"
          f"   [info: canthal span mean {sum(h)/len(h):.1f} px, drift {(max(h)/min(h)-1)*100:.1f}%]")
    return ok


def transfer_report(fd, fo, min_corr, amp_lo, amp_hi):
    n = len(fo["pitch"])
    ok = True
    print(f"[transfer] driver {len(fd['pitch'])} frames -> output {n} frames")
    for ch in ["pitch", "yaw", "roll", "eyeL", "eyeR", "browLi", "browLo", "browRi", "browRo", "lip", "mouthW", "corner"]:
        d = resample([v - fd[ch][0] for v in fd[ch]], n)
        o = [v - fo[ch][0] for v in fo[ch]]
        sd, so = std(d), std(o)
        rng_d = max(d) - min(d)
        rng_o = max(o) - min(o)
        if sd < MOVE_STD[ch]:
            still = so <= max(2 * sd, MOVE_STD[ch])
            print(f"  {ch:6s} driver still (std {sd:.4f}, range {rng_d:.3f}); output std {so:.4f} -> {'PASS' if still else 'FAIL (output moves on its own)'}")
            ok &= still
            continue
        r = pearson(d, o)
        amp = so / sd
        good = (not math.isnan(r)) and r >= min_corr and amp_lo <= amp <= amp_hi
        print(f"  {ch:6s} corr {r:+.3f}  amp {amp:.2f}  range driver {rng_d:.3f} output {rng_o:.3f} -> {'PASS' if good else 'FAIL'}")
        ok &= good
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("driver_csv")
    ap.add_argument("output_csv", nargs="?")
    ap.add_argument("--scale-slack", type=float, default=1.15,
                    help="output extractor-scale range may be at most this times the driver's")
    ap.add_argument("--min-corr", type=float, default=0.8)
    ap.add_argument("--amp-lo", type=float, default=0.6)
    ap.add_argument("--amp-hi", type=float, default=1.6)
    ap.add_argument("--list-steps", action="store_true", help="print every output frame's scale and canthal span")
    a = ap.parse_args()

    drv, _ = load(a.driver_csv)
    brows_d = brow_indices(landmarks(drv, 0))
    fd = features(drv, brows_d)
    print(f"brow indices L in/out {brows_d[0]} R in/out {brows_d[1]}")
    size_report("driver", fd, drv)
    ok_all = True
    if a.output_csv:
        out, names = load(a.output_csv)
        fo = features(out, brow_indices(landmarks(out, 0)))
        ok_size = size_report("output", fo, out, scale_range_pct(drv), a.scale_slack)
        if a.list_steps:
            for i in range(len(names)):
                print(f"    {names[i]}: scale {out['scale'][i]:.4f}  canthal {fo['iod'][i]:.1f}")
        ok_tr = transfer_report(fd, fo, a.min_corr, a.amp_lo, a.amp_hi)
        ok_all = ok_size and ok_tr
    print("RESULT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
