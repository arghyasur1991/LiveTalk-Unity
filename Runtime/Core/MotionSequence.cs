using System;
using System.Collections.Generic;
using UnityEngine;

namespace LiveTalk.Core
{
    using Utils;

    /// <summary>
    /// Edits a driving-motion sequence before it is rendered. LivePortrait
    /// renders every frame from a <see cref="MotionInfo"/> — pose angles,
    /// translation, scale and expression deltas extracted from the driving
    /// frame — so retiming and looping the <i>motion</i> is exact and costs
    /// nothing per frame, where doing the same to the rendered pixels would
    /// blur or ghost. Two operations:
    ///
    /// <list type="bullet">
    /// <item><see cref="Resample"/> retimes the sequence from the clip's
    /// native frame rate to a canonical one (linear interpolation of every
    /// parameter; see the rotation note there).</item>
    /// <item><see cref="MakeLoopable"/> crossfades the tail of the sequence
    /// into its head so the last motion runs straight into the first, in
    /// value and in velocity, and a forward loop has no seam.</item>
    /// </list>
    /// </summary>
    internal static class MotionSequence
    {
        /// <summary>
        /// Frames <see cref="Resample"/> yields for <paramref name="count"/>
        /// source frames: the last source frame is the last one still inside
        /// the clip. Returns <paramref name="count"/> unchanged when either
        /// rate is not positive.
        /// </summary>
        public static int ResampledCount(int count, float sourceFps, float targetFps)
        {
            if (count <= 1 || sourceFps <= 0f || targetFps <= 0f || Mathf.Approximately(sourceFps, targetFps))
                return count;
            return Mathf.FloorToInt((count - 1) * (targetFps / sourceFps) + 1e-4f) + 1;
        }

        /// <summary>
        /// Frames the edited sequence has after resampling and looping:
        /// <see cref="ResampledCount"/> minus the loop blend window.
        /// </summary>
        public static int EditedCount(int count, float sourceFps, float targetFps, int loopBlendFrames)
        {
            int resampled = ResampledCount(count, sourceFps, targetFps);
            int blend = ClampBlend(resampled, loopBlendFrames);
            return resampled - blend;
        }

        /// <summary>
        /// Retimes <paramref name="motions"/> from <paramref name="sourceFps"/>
        /// to <paramref name="targetFps"/>. Output frame <c>i</c> sits at
        /// source time <c>i * sourceFps / targetFps</c> and is the linear blend
        /// of the two source frames around it. Returns the input list itself
        /// when no resampling is needed.
        /// </summary>
        public static List<MotionInfo> Resample(List<MotionInfo> motions, float sourceFps, float targetFps)
        {
            if (motions == null) throw new ArgumentNullException(nameof(motions));
            int count = ResampledCount(motions.Count, sourceFps, targetFps);
            if (count == motions.Count)
                return motions;

            float step = sourceFps / targetFps;
            var result = new List<MotionInfo>(count);
            for (int i = 0; i < count; i++)
            {
                float t = i * step;
                int a = Mathf.Min(Mathf.FloorToInt(t), motions.Count - 1);
                int b = Mathf.Min(a + 1, motions.Count - 1);
                float w = Mathf.Clamp01(t - a);
                result.Add(Lerp(motions[a], motions[b], w));
            }
            return result;
        }

        /// <summary>
        /// Makes a forward loop of <paramref name="motions"/> seamless. The
        /// result has <c>len - N</c> frames (<c>N = blendFrames</c>): the pure
        /// run <c>S[N .. len-N-1]</c>, then <c>N</c> frames in which the
        /// original tail <c>S[len-N+k]</c> is crossfaded toward the original
        /// head <c>S[k]</c> with a smoothstep weight. The blended run ends
        /// exactly on <c>S[N-1]</c>, whose successor is <c>S[N]</c> — the first
        /// frame of the result — so the wrap is one ordinary frame step, and
        /// because smoothstep has zero slope at both ends the velocity is
        /// continuous into and out of the window too. Nothing is reversed.
        ///
        /// The first <c>N</c> source frames are consumed as blend targets rather
        /// than played on their own, which is why the loop is shorter than the
        /// clip: a loop of the full length would have to invent the frame
        /// before <c>S[0]</c>. Returns the input list itself when
        /// <paramref name="blendFrames"/> is not positive or the sequence is
        /// too short to blend.
        /// </summary>
        public static List<MotionInfo> MakeLoopable(List<MotionInfo> motions, int blendFrames)
        {
            if (motions == null) throw new ArgumentNullException(nameof(motions));
            int n = ClampBlend(motions.Count, blendFrames);
            if (n <= 0)
                return motions;

            int len = motions.Count;
            var result = new List<MotionInfo>(len - n);
            for (int i = n; i < len - n; i++)
                result.Add(motions[i]);
            for (int k = 0; k < n; k++)
            {
                float w = n == 1 ? 1f : SmoothStep((float)k / (n - 1));
                result.Add(Lerp(motions[len - n + k], motions[k], w));
            }
            return result;
        }

        /// <summary>
        /// Linear blend of two motions. Pose is interpolated per Euler angle
        /// (pitch, yaw, roll in degrees, the form the motion extractor emits
        /// and <see cref="MathUtils.GetRotationMatrix"/> consumes). For head
        /// poses — a few tens of degrees at most, and the two motions being
        /// blended are neighbouring frames or a tail and head of the same
        /// idle clip — the difference from a slerp is second order and far
        /// below what the renderer resolves, so the angles are blended
        /// directly and the matrix is rebuilt from them.
        /// </summary>
        public static MotionInfo Lerp(MotionInfo a, MotionInfo b, float t)
        {
            if (t <= 0f) return Clone(a);
            if (t >= 1f) return Clone(b);
            var m = new MotionInfo
            {
                Pitch = LerpArray(a.Pitch, b.Pitch, t),
                Yaw = LerpArray(a.Yaw, b.Yaw, t),
                Roll = LerpArray(a.Roll, b.Roll, t),
                Translation = LerpArray(a.Translation, b.Translation, t),
                Expression = LerpArray(a.Expression, b.Expression, t),
                Scale = LerpArray(a.Scale, b.Scale, t),
                Keypoints = LerpArray(a.Keypoints, b.Keypoints, t),
            };
            m.RotationMatrix = MathUtils.GetRotationMatrix(m.Pitch, m.Yaw, m.Roll);
            return m;
        }

        private static MotionInfo Clone(MotionInfo a)
        {
            var m = new MotionInfo
            {
                Pitch = (float[])a.Pitch?.Clone(),
                Yaw = (float[])a.Yaw?.Clone(),
                Roll = (float[])a.Roll?.Clone(),
                Translation = (float[])a.Translation?.Clone(),
                Expression = (float[])a.Expression?.Clone(),
                Scale = (float[])a.Scale?.Clone(),
                Keypoints = (float[])a.Keypoints?.Clone(),
            };
            m.RotationMatrix = a.RotationMatrix ?? MathUtils.GetRotationMatrix(m.Pitch, m.Yaw, m.Roll);
            return m;
        }

        private static float[] LerpArray(float[] a, float[] b, float t)
        {
            if (a == null) return (float[])b?.Clone();
            if (b == null) return (float[])a.Clone();
            int n = Math.Min(a.Length, b.Length);
            var r = new float[n];
            for (int i = 0; i < n; i++)
                r[i] = a[i] + (b[i] - a[i]) * t;
            return r;
        }

        private static float SmoothStep(float x)
        {
            x = Mathf.Clamp01(x);
            return x * x * (3f - 2f * x);
        }

        /// <summary>A blend window must leave at least one pure frame: <c>2N &lt; len</c>.</summary>
        private static int ClampBlend(int count, int blendFrames)
        {
            if (blendFrames <= 0 || count < 3)
                return 0;
            return Mathf.Min(blendFrames, (count - 1) / 2);
        }
    }
}
