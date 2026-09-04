namespace LiveTalk.API
{
    /// <summary>
    /// How the driving motion is edited before LivePortrait renders it.
    /// The motion extracted from each driving frame (pose, translation, scale,
    /// expression) is retimed and looped in keypoint space, then rendered —
    /// so the output frames are exact renders, not blended pixels.
    /// </summary>
    public sealed class DrivingMotionOptions
    {
        /// <summary>
        /// Frame rate the output is retimed to. 0 keeps the driving clip's
        /// native rate (no resampling). Default 25, the rate the lip-sync
        /// frames are generated at, so idle and speech share one clock.
        /// </summary>
        public float TargetFps { get; set; } = 25f;

        /// <summary>
        /// Length of the crossfade that makes a forward loop seamless, in
        /// seconds of output. 0 disables looping. Default 0.4 s (10 frames at
        /// 25 fps). The loop is shorter than the clip by this much; see
        /// <c>MotionSequence.MakeLoopable</c>.
        /// </summary>
        public float LoopBlendSeconds { get; set; } = 0.4f;

        /// <summary>
        /// Native frame rate of the driving frames. 0 means "read it from the
        /// clip" when a <see cref="UnityEngine.Video.VideoPlayer"/> is the
        /// source, and "unknown, do not resample" otherwise.
        /// </summary>
        public float SourceFps { get; set; }

        /// <summary>True when <see cref="LoopBlendSeconds"/> asks for a loopable result.</summary>
        public bool Loopable => LoopBlendSeconds > 0f;

        /// <summary>
        /// Multiplier on the driving clip's expression change relative to its
        /// first frame, applied before rendering (the analogue of upstream
        /// LivePortrait's <c>driving_multiplier</c>). 1 reproduces the driver
        /// as extracted. LivePortrait transfers expression conservatively, and
        /// a source whose resting face is already expressive — a smiling
        /// portrait, say — absorbs part of any opposite-signed delta, so a
        /// clearly sad driver can arrive as a faint frown. Values of 1.3–1.7
        /// restore legibility; above ~2 the mouth and brows start to tear.
        /// Head pose, translation and scale are not affected.
        /// </summary>
        public float ExpressionGain { get; set; } = 1f;

        /// <summary>Frames in the loop blend window at the effective output rate.</summary>
        internal int LoopBlendFrames(float outputFps) =>
            LoopBlendSeconds > 0f && outputFps > 0f ? UnityEngine.Mathf.Max(1, UnityEngine.Mathf.RoundToInt(LoopBlendSeconds * outputFps)) : 0;

        /// <summary>The rate the edited sequence plays at: <see cref="TargetFps"/>, or the source rate when not resampling.</summary>
        internal float OutputFps => TargetFps > 0f ? TargetFps : SourceFps;

        /// <summary>The defaults: 25 fps, 0.4 s loop blend.</summary>
        public static DrivingMotionOptions Default => new();
    }
}
