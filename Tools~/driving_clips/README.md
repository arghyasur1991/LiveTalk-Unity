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

The shipped clips come from exactly this
pipeline: `mblab_build.py` → `mblab_rich.py` → `render_clips.py` with the
table in `clips.py`.

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
     directory — re-point `iris_color.png` / `iris_bump.png` at load
     (`lp_scene.repath_images`);
  4. start from an empty homefile when running headless.
- `ffmpeg` for assembling the mp4; Python 3 with Pillow + numpy for
  `analyze.py` / `sheet.py`.

## Scripts

| Script | Does |
|---|---|
| `mblab_install.py` | Download + install + enable MB-Lab into the user addons dir |
| `mblab_build.py` | Create and finalize a realistic character (skin shader, 83 expression shape keys), save `mblab_char.blend` |
| `lp_scene.py` | Shared: open + repath the character, measure the head, camera / lights / backdrop / render settings (512×512, 25 fps, 85 mm, grey) |
| `mblab_rich.py` | Dress the character: eyebrows, eyelashes and a combed short hairstyle as hair Curves that follow the skin, subsurface skin, wet cornea, crew-neck top. `--test` renders five poses, `--flicker N` renders a slow yaw for a stability check |
| `sk_atlas.py` | One still per shape key / head axis so a key's meaning and sign can be read before authoring |
| `clips.py` | **The seven clips as data** — every timing, angle and amplitude |
| `render_clips.py` | Bake a clip from the table (procedural layers + gesture curves), render the PNG sequence, encode the mp4 |
| `analyze.py` | Consecutive-frame MAD, first-vs-last MAD, background MAD at the expression peak, contact sheet |
| `sheet.py` | Contact sheet of PNGs with optional crop / scale / labels |

## Building the clips

```bash
B=/Applications/Blender.app/Contents/MacOS/Blender
$B --background --python mblab_install.py
$B --background --python mblab_build.py                       # -> mblab_char.blend (edit the save path)
$B --background --python mblab_rich.py -- --src work/mblab_char.blend \
     --out work/mblab_char_rich.blend --test work/rich_test --flicker 10
$B --background --python sk_atlas.py -- --blend work/mblab_char_rich.blend --out work/atlas
$B --background --python render_clips.py -- --blend work/mblab_char_rich.blend \
     --frames work/frames --mp4 ../../Resources/driving            # all seven clips
$B --background --python render_clips.py -- --blend work/mblab_char_rich.blend \
     --frames work/preview --stride 25 --samples 16 --clips smile  # quick look, every 25th frame
python analyze.py smile=work/frames/smile
```

`render_clips.py` renders ~2.5 s/frame on an M-series Mac at 32 TAA
samples (the hair dominates); the full set (~1400 frames) is about an hour.
`--stride N --samples 16` gives a contact-sheet preview in a couple of
minutes. Frames are PNG; the mp4 is H.264 High / yuv420p / 25 fps, the same
container family as the stock clips so Unity's `VideoClip` import is
unchanged.

## Editing a clip

Everything is in `clips.py`. A clip row has `seconds`, `seed`, optional
overrides for the procedural layers (`breath`, `sway`, `blinks`,
`saccades`, `micro`, `asym`), `beats` (instants a blink must not land on)
and `gestures`: per-channel breakpoint lists `[(t, value), ...]` in
seconds. Channels and their signs are documented at the top of the file
and were verified with `sk_atlas.py`; angles are degrees on the head bone
(35 % of every head rotation is carried by the neck bone). Curves are
monotone cubics, so a breakpoint list never overshoots and every gesture
eases in and out. All procedural layers are multiplied by an envelope that
is zero at both ends of the clip, so **frame 0 and the last frame are the
identical rest pose** — across all seven clips.

## Authoring rules for a clip

- **512x512, 25 fps**, face filling ~55–65 % of the frame, plain neutral
  background, soft even lighting. Match the framing of the existing clips
  (`lp_scene.py` owns it; do not change it per clip).
- **Start and end at the same rest pose**, eased to zero velocity.
  Avatar creation wraps last→first without a motion-space blend; a clip
  that already returns to rest loops, and a shared rest pose makes cuts
  between expression clips seamless.
- Idle: 25–30 s, aperiodic blinks (never on a fixed period, never on a
  gesture beat), slow breathing with one deeper breath, small gaze shifts
  and head sway, a few tilts/turns ≤ 6°, no repeated gesture. Mouth stays
  relaxed and closed (lip-sync regenerates it during speech).
- Expressions (`approve`, `confused`, `disapprove`, `sad`, `smile`,
  `surprised`): rest → expression → rest, 3–6 s, breathing and sway kept
  running underneath.
- **Key brows at ≥ 100 %** — brow raise transfers weaker than authored
  (the brow channels may go to 1.5; the renderer raises the keys' range).
- **Never blink while the eyes are in an extreme state** (held wide, gaze
  lowered). LivePortrait pops the whole face for a few frames when a
  driver blink starts from wide-open or lowered eyes (measured 5.5–8.1
  consecutive MAD against a 0.4 median); use `blinks.forced` to put the
  blink in the return instead.
- Nothing symmetric: `asym` scales left/right brows, smile, squint, cheeks.
- **The last blink must be over before the clip's final 0.3 s.** A blink
  takes ~0.45 s to close and reopen; one at 3.05 s in a 3.84 s clip left the
  final frame with half-shut eyes and a first↔last difference of 1.5 in the
  *encoded* clip. Check the mp4, not the PNGs.
- **Emotions need to be over-driven, not just brows.** LivePortrait is
  conservative on everything subtle: sad reads only with inner brows at 1.5,
  outer brows down ~0.9, mouth corners over-driven to -1.6 with the mouth
  narrowed (`wide` -0.55) and the lower lip out 0.75, gaze -0.45 and ~7° of
  pitch, held for 2–3 s. The first two passes (smile -0.7 then -0.95)
  measured a mouth-corner drop of 0.05 IOD on the *driver* — invisible after
  transfer. `smile`, `lowerOut` and `wide` may exceed 1.0 (`OVERDRIVE`).
- Eyes stay on the camera: the renderer counter-rotates the gaze against
  head yaw/pitch (`saccades.vor`), so a nod or turn does not read as
  looking away.

## Validating a clip

`analyze.py` checks hold / wrap / border MAD (loop seams and hair shimmer).
First and last frames of a clip should match so idle wraps.

## Character notes (what `mblab_rich.py` fixes and why)

- **Render with Eevee Next** (`lp_scene.setup_render` default), ray-traced
  AO/contact shading on, `AgX - High Contrast`, a saturated **green**
  backdrop and a small hard key high on the key side (80 W, 0.45 m) with a
  ~5:1 fill and a top light — the driver is read by a motion extractor,
  not a person, so what matters is separable features: nose and
  nasolabial shadows, defined lips, brows and lashes against skin, skin
  against background. Cycles (`engine="CYCLES"`: random-walk SSS, HDR
  reflections, DOF) is kept as an option; its extra realism did not
  change what the extractor read and it is ~4x slower. ~3 s/frame at 16
  TAA samples on Apple Silicon. Do not run a Unity LivePortrait pass at the
  same time: Metal contention silently killed a render at frame 0.
- **The painted brows are retouched out of the skin map** (`paint_albedo`).
  MB-Lab's albedo has eyebrows painted 9–14.5 mm above the eye line; the
  hair brows sat at 16.5 mm, so a thin red-brown line showed under each
  brow — a second edge for the extractor. The band is found by luminance on
  a grid of skin hits, filled with the skin colour above it, and the hair
  brows are grown as one smooth arc on the measured ridge (per-column
  snapping to the band split each brow into two rows). The same pass
  saturates and darkens the lips (found by redness against the cheeks) so
  the mouth reads as a feature; the retouched map is packed into the blend.
- **Hair is a short combed-back cut**, not long strands. Long procedural
  hair read as straw and swung past the face edges — the extractor then sees
  hair, not face, changing frame to frame. The head-hair material is matte
  (roughness 0.58) because bright strand highlights flicker between frames.
- **Lashes are rooted through the skin's UV island.** `Deform Curves on
  Surface` finds a root by sampling the surface at the strand's UV; MB-Lab's
  eyelash cards are stacks of planes sharing UV space, so the lookup missed
  and the lashes stood still while the lids closed (the cards themselves move
  13 mm under `eyeClosed`). Margin positions still come from the cards; the
  UV comes from the nearest skin face, interpolated.

- **Hair Curves, not particle hair.** `ParticleHairKey.co_object_set`
  writes are lost on save in background mode (the original particle
  system never receives them). Curves store their points and ride the
  skin through *Deform Curves on Surface* (surface = body, `rest_position`
  attribute + `surface_uv_coordinate` per curve). Strands are authored
  procedurally: forehead swept back over the crown, crown outward and
  down, sides/back down with a skull-clamp so nothing pokes through.
- **Strand width matters for flicker.** Strands thinner than a pixel
  shimmer between frames; roots ≈ 0.5 mm, tips ≈ 0.18 mm, a 1.8 px film
  filter and 32–48 TAA samples bring the hair region's frame-to-frame MAD
  under a slow yaw to ~2/255 with no pops. A static hold is bit-identical.
- **Skin displacement.** The skin material's *Displacement* output (2048
  map, rendered as bump by EEVEE) leaves blocky red patches at the
  nostrils and eye corners; the Displace *modifier* already applies the
  map to geometry, so the shader-side link is removed.
- **Eyelash cards stay `BLENDED`.** Hashed alpha leaves pink speckle under
  the eyes.
- **Brows are hairs on the skin**, placed by ray cast relative to the eye
  centres at the height `paint_albedo` measures for the painted brow, so
  they rise and knit with the brow shape keys.
- The body's own displacement moves skin ±5 mm; the top sits 11 mm out or
  the skin pokes through in blobs.
