"""The seven driving clips as data.  Every number the renderer uses lives
here; ``render_clips.py`` turns a row into keyframes and frames.

Conventions (rich MB-Lab character, see ``sk_atlas.py`` renders):

* Angles in degrees on the ``head`` bone.  ``pitch`` + = chin down,
  ``yaw`` + = turn toward the character's right (camera left),
  ``roll`` + = top of the head toward her left shoulder (camera right).
* Expression channels are signed in [-1, 1] and map onto MB-Lab
  ``_max`` / ``_min`` shape-key pairs (``render_clips.CHANNELS``):
  ``browL/browR`` + up / - down, ``browMid`` + inner brows up / - furrow,
  ``squeezeL/R`` + brows pulled in, ``eyeL/eyeR`` + closed / - wide,
  ``squintL/R`` lower lid up, ``cheekL/R`` cheek raise, ``smile`` + corners
  up / - corners down, ``smileL/R`` one corner, ``open`` jaw open,
  ``press`` + lips pressed / - lips parted, ``wide`` + mouth narrowed /
  - widened, ``lowerOut`` + lower lip out / - in, ``gazeH`` + eyes toward
  her left (camera right, the way +yaw turns the head), ``gazeV`` + eyes
  up (1.0 ~ 36 deg), ``chest`` chest expansion,
  ``swallow`` deglutition, ``nostrils`` flare.
* Gesture curves are absolute-time breakpoints ``[(seconds, value), ...]``
  per channel, interpolated with a monotone cubic (no overshoot, continuous
  velocity).  Every curve implicitly starts and ends at 0.

Procedural layers (all windowed to zero at both ends of the clip):

* ``breath``: chest expansion + a few mm of torso rise + a whisper of head
  pitch at ``bpm``; ``deep_at`` adds one slower, deeper breath.
* ``sway``: incommensurate sines + smooth noise on head pitch/yaw/roll and
  a little on the spine.  Amplitudes in degrees.
* ``blinks``: aperiodic — intervals drawn from ``interval`` seconds, never
  within ``avoid`` of a listed beat (gesture peaks); close ``close_frames``,
  open ``open_frames``, a 1-2 deg eye dart just before each.
* ``saccades``: gaze re-targets every ``interval`` seconds by up to
  ``max_deg``; the eyes also counter-rotate against head yaw (``vor``) so
  she keeps looking at the camera.
* ``micro``: slow low-amplitude noise on brows and mouth corners.
"""

FPS = 25

# Shared defaults; a clip overrides what it needs.
DEFAULTS = dict(
    envelope=0.8,                        # seconds of ease from/to rest at each end
    breath=dict(bpm=13.0, chest=0.55, rise=0.0045, pitch=-0.35, shoulder=0.6, deep_at=[], deep_gain=1.8),
    sway=dict(yaw=1.2, pitch=0.7, roll=0.55, spine=0.25, periods=(7.3, 11.1, 5.7), noise=0.5),
    blinks=dict(interval=(2.0, 6.0), avoid=0.35, close_frames=3, open_frames=5, dart=1.5, edge=0.7,
                asym=(1.0, 0.96), brow_drop=0.06),
    saccades=dict(interval=(1.0, 3.0), max_deg=2.0, move_frames=3, vor=0.8, vor_pitch=0.9),
    micro=dict(brow=0.02, mouth=0.015, period=3.5),
    # left/right scale so nothing is ever perfectly symmetric
    asym=dict(brow=(1.0, 0.92), smile=(1.0, 0.9), squint=(0.94, 1.0), cheek=(1.0, 0.9)),
    beats=[],                            # blink-free instants (gesture peaks)
    gestures={},
)

CLIPS = {
    # -------------------------------------------------------------- idle
    "talk-neutral": dict(
        seconds=27.0, seed=101,
        breath=dict(deep_at=[13.2]),
        sway=dict(yaw=1.5, pitch=0.9, roll=0.7),
        blinks=dict(interval=(2.2, 5.5)),
        beats=[4.2, 9.6, 15.4, 21.0],
        gestures={
            # four small tilts / turns (<= 6 deg), each with its own timing
            "yaw":   [(3.2, 0), (4.2, -5.0), (5.6, -4.4), (7.0, 0),
                      (14.6, 0), (15.4, 4.6), (16.9, 4.0), (18.4, 0)],
            "roll":  [(8.4, 0), (9.6, 5.5), (11.4, 5.0), (12.9, 0),
                      (20.0, 0), (21.0, -4.2), (22.2, -3.8), (23.5, 0)],
            # two near-imperceptible micro-nods
            "pitch": [(6.0, 0), (6.35, 1.6), (6.8, 0), (19.2, 0), (19.5, 1.3), (19.95, 0.2), (20.3, 0)],
            # one swallow, mouth stays relaxed
            "swallow": [(10.6, 0), (11.0, 0.8), (11.5, 0)],
            "press": [(2.5, 0), (2.9, -0.06), (3.6, 0), (17.0, 0), (17.5, -0.05), (18.3, 0)],
        },
    ),
    # -------------------------------------------------------------- approve
    "approve": dict(
        seconds=4.6, seed=202,
        beats=[1.05, 1.85],
        gestures={
            "pitch":  [(0.55, 0), (1.05, 7.5), (1.45, 1.5), (1.85, 5.5), (2.35, 0.8), (2.9, 0)],
            "browL":  [(0.5, 0), (0.95, 0.35), (1.6, 0.1), (2.6, 0)],
            "browR":  [(0.5, 0), (0.95, 0.30), (1.6, 0.08), (2.6, 0)],
            "smile":  [(0.8, 0), (1.7, 0.42), (2.6, 0.42), (3.7, 0)],
            "smileL": [(0.9, 0), (1.8, 0.12), (2.6, 0.12), (3.6, 0)],
            "cheekL": [(0.9, 0), (1.8, 0.12), (2.6, 0.1), (3.6, 0)],
            "cheekR": [(0.9, 0), (1.8, 0.10), (2.6, 0.08), (3.6, 0)],
            "squintL": [(1.0, 0), (1.8, 0.18), (2.6, 0.15), (3.6, 0)],
            "squintR": [(1.0, 0), (1.8, 0.16), (2.6, 0.13), (3.6, 0)],
        },
    ),
    # -------------------------------------------------------------- confused
    "confused": dict(
        seconds=4.8, seed=303,
        beats=[1.3],
        gestures={
            "roll":    [(0.5, 0), (1.3, 7.0), (2.4, 7.0), (3.6, 0)],
            "yaw":     [(0.5, 0), (1.3, 3.0), (2.4, 2.8), (3.6, 0)],
            "pitch":   [(0.5, 0), (1.3, 1.5), (2.4, 1.5), (3.6, 0)],
            "browL":   [(0.55, 0), (1.25, 1.0), (2.4, 1.0), (3.5, 0)],
            "browMid": [(0.55, 0), (1.25, 0.35), (2.4, 0.35), (3.5, 0)],
            "browR":   [(0.55, 0), (1.25, -0.7), (2.4, -0.7), (3.5, 0)],
            "squeezeR": [(0.55, 0), (1.25, 0.65), (2.4, 0.65), (3.5, 0)],
            "squintL": [(0.6, 0), (1.3, 0.22), (2.4, 0.22), (3.5, 0)],
            "squintR": [(0.6, 0), (1.3, 0.32), (2.4, 0.32), (3.5, 0)],
            "wide":    [(0.7, 0), (1.4, 0.35), (2.4, 0.35), (3.5, 0)],
            "press":   [(0.7, 0), (1.4, 0.22), (2.4, 0.22), (3.5, 0)],
            "gazeH":   [(0.6, 0), (1.3, -0.06), (2.4, -0.06), (3.4, 0)],
        },
    ),
    # -------------------------------------------------------------- disapprove
    "disapprove": dict(
        seconds=4.8, seed=404,
        beats=[1.0, 1.55, 2.1, 2.65],
        gestures={
            "yaw":     [(0.55, 0), (1.0, -5.0), (1.55, 4.8), (2.1, -4.2), (2.65, 3.2), (3.2, 0)],
            "pitch":   [(0.55, 0), (1.2, 1.2), (2.7, 1.2), (3.4, 0)],
            "browL":   [(0.5, 0), (1.1, -0.6), (2.8, -0.6), (3.7, 0)],
            "browR":   [(0.5, 0), (1.1, -0.55), (2.8, -0.55), (3.7, 0)],
            "browMid": [(0.5, 0), (1.1, -0.7), (2.8, -0.7), (3.7, 0)],
            "squeezeL": [(0.5, 0), (1.1, 0.7), (2.8, 0.7), (3.7, 0)],
            "squeezeR": [(0.5, 0), (1.1, 0.65), (2.8, 0.65), (3.7, 0)],
            "press":   [(0.6, 0), (1.2, 0.55), (2.8, 0.5), (3.7, 0)],
            "smile":   [(0.6, 0), (1.2, -0.35), (2.8, -0.35), (3.7, 0)],
            "squintL": [(0.6, 0), (1.2, 0.15), (2.8, 0.15), (3.6, 0)],
            "squintR": [(0.6, 0), (1.2, 0.18), (2.8, 0.18), (3.6, 0)],
        },
    ),
    # -------------------------------------------------------------- sad
    "sad": dict(
        seconds=5.6, seed=505,
        beats=[1.6],
        breath=dict(bpm=11.0),
        gestures={
            "gazeV":   [(0.5, 0), (1.5, -0.30), (3.6, -0.30), (4.9, 0)],
            "pitch":   [(0.5, 0), (1.7, 4.5), (3.6, 4.5), (5.0, 0)],
            "roll":    [(0.6, 0), (1.8, 2.5), (3.6, 2.5), (5.0, 0)],
            "browMid": [(0.5, 0), (1.6, 1.0), (3.7, 1.0), (4.9, 0)],
            "browL":   [(0.5, 0), (1.6, -0.35), (3.7, -0.35), (4.9, 0)],
            "browR":   [(0.5, 0), (1.6, -0.30), (3.7, -0.30), (4.9, 0)],
            "smile":   [(0.6, 0), (1.7, -0.7), (3.7, -0.7), (4.9, 0)],
            "lowerOut": [(0.6, 0), (1.7, 0.25), (3.7, 0.25), (4.9, 0)],
            "press":   [(0.6, 0), (1.7, 0.15), (3.7, 0.15), (4.9, 0)],
            "eyeL":    [(0.6, 0), (1.7, 0.12), (3.7, 0.12), (4.9, 0)],
            "eyeR":    [(0.6, 0), (1.7, 0.10), (3.7, 0.10), (4.9, 0)],
            # slow exhale: chest empties, shoulders drop
            "chest":   [(1.0, 0), (1.8, 0.5), (3.4, -0.2), (4.6, 0)],
            "rise":    [(1.0, 0), (1.8, 0.004), (3.4, -0.006), (4.8, 0)],
        },
    ),
    # -------------------------------------------------------------- smile
    "smile": dict(
        seconds=4.8, seed=606,
        beats=[2.0],
        # one deliberate blink right after the peak
        blinks=dict(interval=(9.0, 9.0), forced=[2.35]),
        gestures={
            "smile":   [(0.5, 0), (1.3, 0.45), (2.0, 0.8), (3.0, 0.75), (4.0, 0)],
            "smileL":  [(0.5, 0), (1.3, 0.08), (2.0, 0.18), (3.0, 0.16), (4.0, 0)],
            "cheekL":  [(0.6, 0), (2.0, 0.3), (3.0, 0.28), (4.0, 0)],
            "cheekR":  [(0.6, 0), (2.0, 0.26), (3.0, 0.24), (4.0, 0)],
            "squintL": [(0.6, 0), (2.0, 0.5), (3.0, 0.45), (4.0, 0)],
            "squintR": [(0.6, 0), (2.0, 0.55), (3.0, 0.5), (4.0, 0)],
            "press":   [(0.7, 0), (2.0, -0.18), (3.0, -0.16), (4.0, 0)],
            "browL":   [(0.6, 0), (1.8, 0.15), (3.0, 0.1), (4.0, 0)],
            "browR":   [(0.6, 0), (1.8, 0.12), (3.0, 0.08), (4.0, 0)],
            "pitch":   [(0.6, 0), (2.0, -1.8), (3.0, -1.6), (4.1, 0)],
            "roll":    [(0.6, 0), (2.0, 2.2), (3.0, 2.0), (4.1, 0)],
        },
    ),
    # -------------------------------------------------------------- surprised
    "surprised": dict(
        seconds=3.8, seed=707,
        beats=[0.9],
        gestures={
            "browL":   [(0.55, 0), (0.85, 1.0), (1.9, 1.0), (2.9, 0)],
            "browR":   [(0.55, 0), (0.85, 1.0), (1.9, 0.95), (2.9, 0)],
            "browMid": [(0.55, 0), (0.85, 1.0), (1.9, 0.9), (2.9, 0)],
            "eyeL":    [(0.55, 0), (0.85, -0.8), (1.9, -0.7), (2.9, 0)],
            "eyeR":    [(0.55, 0), (0.85, -0.8), (1.9, -0.7), (2.9, 0)],
            "open":    [(0.6, 0), (0.95, 0.38), (1.9, 0.34), (2.9, 0)],
            "press":   [(0.6, 0), (0.95, -0.2), (1.9, -0.2), (2.9, 0)],
            "pitch":   [(0.55, 0), (0.95, -4.5), (1.9, -3.8), (3.0, 0)],
            "yaw":     [(0.55, 0), (0.95, 1.5), (1.9, 1.3), (3.0, 0)],
            "chest":   [(0.55, 0), (0.95, 0.6), (1.9, 0.5), (3.0, 0)],
            "nostrils": [(0.6, 0), (0.95, 0.4), (1.9, 0.3), (2.9, 0)],
        },
    ),
}
