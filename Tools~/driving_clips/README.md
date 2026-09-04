# Driving clips — authoring pipeline

The clips in `Resources/driving/*.mp4` are the motion source for every
avatar's idle and expression frames: LivePortrait extracts head pose and
expression from each driving frame and retargets it onto the portrait.
Driver quality is therefore the ceiling on animation quality.

This folder builds those clips from a rendered realistic face in Blender
instead of stock footage. Measured against the stock clips (2026-09-04),
a rendered face is the cleaner driver: output frames are bit-identical
while the face holds still, the clip returns exactly to its rest pose so
it loops, and 10° of yaw transfers with pitch and roll flat.

## Requirements

- Blender 4.5 (`/Applications/Blender.app` on macOS; adjust paths).
- **MB-Lab 1.8.1**, the `animate1978/MB-Lab` release tagged `1_8_1`
  (`bl_info.blender = (4, 0, 0)`). `mblab_install.py` downloads and
  installs it. Four things it needs on 4.5, all handled by the scripts:
  1. enable with `addon_utils.enable(..., default_set=True)` — otherwise
     `remove_censors()` reads a `None` preferences entry;
  2. do not use the addon's "Use EEVEE" toggle (writes the removed
     `BLENDER_EEVEE` enum); leave `mblab_use_cycles=True` so the skin
     material is built, then set `BLENDER_EEVEE_NEXT` yourself;
  3. `humanoid_library.blend` ships iris texture paths into a 4.1 addons
     directory — re-point `iris_color.png` / `iris_bump.png` at load;
  4. start from an empty homefile when running headless.
- `ffmpeg` for assembling the mp4.

## Scripts

| Script | Does |
|---|---|
| `mblab_install.py` | Download + install + enable MB-Lab into the user addons dir |
| `mblab_build.py` | Create and finalize a realistic character (skin shader, 83 expression shape keys), save `mblab_char.blend` |
| `mblab_anim.py` | Camera / lights / backdrop / 512x512 @ 25 fps render setup and the keyframed clip; renders a PNG sequence |
| `analyze.py` | Consecutive-frame and first-vs-last mean absolute differences over a PNG folder (the seam / jitter metrics) |
| `inspect_mats.py` | Dump the character's materials and texture paths (used to find the iris path bug) |

Run each headless:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python mblab_install.py
/Applications/Blender.app/Contents/MacOS/Blender --background --python mblab_build.py
/Applications/Blender.app/Contents/MacOS/Blender --background --python mblab_anim.py
ffmpeg -framerate 25 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p clip.mp4
```

## Authoring rules for a clip

- **512x512, 25 fps**, face filling ~55–65 % of the frame, plain neutral
  background, soft even lighting. Match the framing of the existing clips.
- **Start and end at the same rest pose**, eased to zero velocity. The
  avatar pipeline crossfades the last 0.4 s into the first in keypoint
  space, but a clip that already returns to rest loops best.
- Idle: 20–30 s, aperiodic blinks (never on a fixed period), slow
  breathing, small gaze shifts and head sway; no repeated gesture.
- Expressions (`approve`, `confused`, `disapprove`, `sad`, `smile`,
  `surprised`): rest → expression → rest, 3–6 s.
- **Key brows at ≥ 100 %** — brow raise transfers weaker than authored.
- Give the character hair and geometric eyebrows/lashes; a bald,
  texture-only face extracts less motion.

Validate a new clip by driving a portrait through
`LiveTalkAPI.GenerateAnimatedTexturesAsync(image, framesFolder)` and
running `analyze.py` on the output: hold frames should be ~0.0 apart,
first-vs-last ~0.0, no background change at expression peaks.
