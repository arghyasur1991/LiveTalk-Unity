using System;
using System.Threading.Tasks;
using UnityEngine;

namespace LiveTalk.Utils
{
    using Core;

    /// <summary>
    /// Restores what MuseTalk's 256² decode loses, after the face is blended
    /// back onto its avatar frame. One instance per generated stream; frames
    /// must arrive in order.
    ///
    /// <list type="number">
    /// <item><b>Detail.</b> The decode has no skin texture (stubble, pores),
    /// so the whole masked lower face reads soft. The avatar frame's high-pass
    /// is added back wherever the low-pass of the blended and original faces
    /// still agree. Where the mouth changed shape they disagree and nothing
    /// is added, so the closed source mouth never ghosts over an open one.</item>
    /// <item><b>Stability.</b> Each frame is decoded independently, so the
    /// patch boils. The generated residual (blended minus original) is pulled
    /// toward the previous frame's residual where it barely changed and left
    /// alone where it moved, so articulation keeps its timing. Smoothing the
    /// residual rather than pixels keeps the lower face locked to a moving
    /// head.</item>
    /// </list>
    /// </summary>
    internal sealed class MouthRefiner
    {
        const float DetailSigma = 2f;    // px at avatar resolution: what counts as skin detail
        const float AgreeSigma = 4f;     // low-pass scale for "has the face changed shape here"
        const float AgreeRange = 6f;     // low-pass change (grey levels) over which detail fades out
        const float Hold = 0.9f;         // pull toward the previous residual
        const float MotionRange = 14f;   // residual change (grey levels) treated as real motion

        static readonly float[] DetailKernel = GaussianKernel(DetailSigma);
        static readonly float[] AgreeKernel = GaussianKernel(AgreeSigma);

        short[] _previous;               // residual per channel, full-frame coordinates
        int _width, _height;
        bool _hasPrevious;

        /// <summary>
        /// Refines <paramref name="blended"/> in place inside the mask window.
        /// </summary>
        /// <param name="original">The avatar frame the face was blended onto.</param>
        /// <param name="blended">Output of the mask composite, same size as <paramref name="original"/>.</param>
        /// <param name="mask">The blend mask, placed at the crop box origin.</param>
        /// <param name="cropBox">x, y of the mask window in frame pixels (z, w unused).</param>
        public void Apply(Frame original, Frame blended, Frame mask, Vector4 cropBox)
        {
            int frameW = original.width, frameH = original.height;
            if (blended.width != frameW || blended.height != frameH || mask.data == null) return;

            int maskX = (int)cropBox.x, maskY = (int)cropBox.y;
            int x0 = Mathf.Max(0, maskX), y0 = Mathf.Max(0, maskY);
            int x1 = Mathf.Min(frameW, maskX + mask.width), y1 = Mathf.Min(frameH, maskY + mask.height);
            int w = x1 - x0, h = y1 - y0;
            if (w <= 0 || h <= 0) return;

            if (_previous == null || _width != frameW || _height != frameH)
            {
                _previous = new short[frameW * frameH * 3];
                _width = frameW;
                _height = frameH;
                _hasPrevious = false;
            }

            float[] lumaOriginal = Luma(original, x0, y0, w, h);
            float[] lumaBlended = Luma(blended, x0, y0, w, h);
            float[] detailLow = Blur(lumaOriginal, w, h, DetailKernel);
            float[] agreeOriginal = Blur(lumaOriginal, w, h, AgreeKernel);
            float[] agreeBlended = Blur(lumaBlended, w, h, AgreeKernel);

            byte[] src = original.data, dst = blended.data, alphaMap = mask.data;
            short[] previous = _previous;
            bool hasPrevious = _hasPrevious;
            int maskW = mask.width;

            Parallel.For(0, h, j =>
            {
                int y = y0 + j;
                int maskRow = (y - maskY) * maskW;
                for (int i = 0; i < w; i++)
                {
                    int x = x0 + i;
                    int p = (y * frameW + x) * 3;
                    float alpha = alphaMap[(maskRow + x - maskX) * 3] / 255f;
                    if (alpha <= 0.001f)
                    {
                        previous[p] = previous[p + 1] = previous[p + 2] = 0;
                        continue;
                    }

                    int k = j * w + i;
                    float agree = 1f - Mathf.Abs(agreeBlended[k] - agreeOriginal[k]) / AgreeRange;
                    float detail = agree > 0f ? (lumaOriginal[k] - detailLow[k]) * alpha * agree : 0f;

                    float r = dst[p] + detail - src[p];
                    float g = dst[p + 1] + detail - src[p + 1];
                    float b = dst[p + 2] + detail - src[p + 2];

                    if (hasPrevious)
                    {
                        float pr = previous[p], pg = previous[p + 1], pb = previous[p + 2];
                        float change = Mathf.Abs(0.299f * (r - pr) + 0.587f * (g - pg) + 0.114f * (b - pb));
                        float pull = Hold * alpha * Mathf.Clamp01(1f - change / MotionRange);
                        r += (pr - r) * pull;
                        g += (pg - g) * pull;
                        b += (pb - b) * pull;
                    }

                    previous[p] = (short)Mathf.RoundToInt(r);
                    previous[p + 1] = (short)Mathf.RoundToInt(g);
                    previous[p + 2] = (short)Mathf.RoundToInt(b);
                    dst[p] = ToByte(src[p] + r);
                    dst[p + 1] = ToByte(src[p + 1] + g);
                    dst[p + 2] = ToByte(src[p + 2] + b);
                }
            });
            _hasPrevious = true;
        }

        static byte ToByte(float v) => (byte)(v <= 0f ? 0 : v >= 255f ? 255 : (int)(v + 0.5f));

        static float[] Luma(Frame f, int x0, int y0, int w, int h)
        {
            var luma = new float[w * h];
            byte[] d = f.data;
            int stride = f.width;
            Parallel.For(0, h, j =>
            {
                int row = ((y0 + j) * stride + x0) * 3;
                int o = j * w;
                for (int i = 0; i < w; i++, row += 3)
                    luma[o + i] = 0.299f * d[row] + 0.587f * d[row + 1] + 0.114f * d[row + 2];
            });
            return luma;
        }

        /// <summary>Separable Gaussian, edges clamped.</summary>
        static float[] Blur(float[] src, int w, int h, float[] kernel)
        {
            int radius = kernel.Length / 2;
            var tmp = new float[w * h];
            var dst = new float[w * h];
            Parallel.For(0, h, j =>
            {
                int o = j * w;
                for (int i = 0; i < w; i++)
                {
                    float s = 0f;
                    for (int t = -radius; t <= radius; t++)
                        s += kernel[t + radius] * src[o + Math.Clamp(i + t, 0, w - 1)];
                    tmp[o + i] = s;
                }
            });
            Parallel.For(0, h, j =>
            {
                int o = j * w;
                for (int i = 0; i < w; i++)
                {
                    float s = 0f;
                    for (int t = -radius; t <= radius; t++)
                        s += kernel[t + radius] * tmp[Math.Clamp(j + t, 0, h - 1) * w + i];
                    dst[o + i] = s;
                }
            });
            return dst;
        }

        static float[] GaussianKernel(float sigma)
        {
            int radius = Mathf.CeilToInt(3f * sigma);
            var k = new float[radius * 2 + 1];
            float sum = 0f;
            for (int i = -radius; i <= radius; i++)
                sum += k[i + radius] = Mathf.Exp(-i * i / (2f * sigma * sigma));
            for (int i = 0; i < k.Length; i++) k[i] /= sum;
            return k;
        }
    }
}
