using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace LiveTalk.Core
{
    using Utils;

    /// <summary>
    /// Whisper features for MuseTalk computed on a growing audio prefix, so
    /// lip-sync frames can be generated while the speech is still being
    /// synthesised.
    ///
    /// <para><b>What is exact and what is not.</b> Three things separate a
    /// prefix from the finished clip in <see cref="WhisperModel.ProcessAudio"/>:</para>
    /// <list type="number">
    /// <item>The STFT columns near the end of a prefix read the end reflect
    /// padding. Those are recomputed every update; a column is cached as final
    /// only once its window lies inside the audio heard so far
    /// (<see cref="AudioUtils.MelColumnEndSample"/>), so every final column is
    /// identical to the batch path's.</item>
    /// <item>The per-frame windowing is local: frame <c>i</c> reads encoder
    /// timesteps up to <c>2i + 5</c>, i.e. mel columns up to
    /// <see cref="WhisperModel.LastMelColumnFor"/>. <see cref="SafeFrameCount"/>
    /// only counts frames whose last such column is final, which holds back
    /// a margin of <see cref="MarginSeconds"/> (~146 ms: 80 ms of chunk
    /// lookahead + the conv front end + half an STFT window). On
    /// <see cref="Complete"/> the margin is zero and the count equals the
    /// batch path's.</item>
    /// <item>The log-mel reference (<c>librosa.power_to_db(ref=max)</c>) is
    /// the one global term. It is captured from the first
    /// <see cref="WarmupSeconds"/> of audio and <b>held</b> for the utterance.
    /// Measured on real speech the running max reaches its final value within
    /// the first second on most clips; the worst case is a feature error of
    /// ~0.05 on [-1, 1] when the loudest moment comes after the warm-up. A
    /// fixed constant reference is 3–4x worse and is deliberately not used.</item>
    /// </list>
    /// <para>The encoder's self-attention sees the whole 30 s window, so an
    /// encoder run over a prefix is not bit-identical to one over the finished
    /// clip even for early timesteps; that residual is what the frame
    /// equivalence measurement in the changelog quantifies.</para>
    ///
    /// <para><b>Threads.</b> <see cref="Append"/>, <see cref="Complete"/> and
    /// <see cref="Fail"/> may be called from any thread (the TTS chunk relay
    /// calls them on the main thread). <see cref="UpdateAsync"/> runs the
    /// heavy work on a worker and must not be called concurrently with itself;
    /// the generator coroutine that owns this object serialises it.</para>
    /// </summary>
    internal sealed class StreamingAudioFeatures : IDisposable
    {
        /// <summary>
        /// Seconds of audio the log-mel reference is captured from before it
        /// is frozen for the rest of the utterance. Frames are not emitted
        /// before this much audio has arrived (or the utterance has ended).
        /// Trade-off: earlier capture means an earlier first frame but a
        /// reference more likely to be beaten by a later, louder moment;
        /// 0.5 s measured as ~0.05 worst-case feature error on [-1, 1].
        /// </summary>
        internal const float DefaultWarmupSeconds = 0.5f;

        private readonly WhisperModel _whisper;
        private readonly float _warmupSeconds;
        private readonly float _extraContextSeconds;
        private readonly bool _raiseReference;
        private readonly object _lock = new();

        // Input side (any thread, under _lock).
        private float[] _sourcePcm = new float[16000 * 8];
        private int _sourceCount;
        private int _sourceRate;
        private bool _complete;
        private Exception _error;
        private int _consumedSourceCount;

        // Compute side (owned by UpdateAsync).
        private readonly List<float[]> _finalColumns = new();   // mel power, one float[N_MELS] per final column
        private float[] _windowScratch;
        private float _reference;
        private bool _referenceFrozen;
        private WhisperModel.Encoding _encoding;
        private int _encodedLength16k;
        private int _actualTimesteps;
        private int _safeFrameCount;
        private int _totalFrameCount = -1;
        private bool _sessionStarted;
        private bool _disposed;

        /// <param name="whisper">The encoder to run.</param>
        /// <param name="warmupSeconds">See <see cref="DefaultWarmupSeconds"/>.</param>
        /// <param name="extraContextSeconds">
        /// Audio held back beyond the exact local margin before a frame counts
        /// as safe. The encoder's self-attention reaches the whole window, so
        /// a frame encoded with more real audio after it is closer to the
        /// batch result; this trades first-frame latency for that fidelity.
        /// </param>
        /// <param name="raiseReference">
        /// True: the log-mel reference is the running maximum — captured at
        /// warm-up and raised whenever a later, louder moment arrives, so
        /// every frame after the utterance's true peak gets exactly the batch
        /// normalisation (a frame already emitted is never recomputed). False:
        /// frozen at warm-up for the whole utterance.
        /// </param>
        public StreamingAudioFeatures(WhisperModel whisper, float warmupSeconds = DefaultWarmupSeconds,
            float extraContextSeconds = 0f, bool raiseReference = true)
        {
            _whisper = whisper ?? throw new ArgumentNullException(nameof(whisper));
            _warmupSeconds = Mathf.Max(0f, warmupSeconds);
            _extraContextSeconds = Mathf.Max(0f, extraContextSeconds);
            _raiseReference = raiseReference;
        }

        #region Input side

        /// <summary>Appends newly generated PCM. All chunks must share one sample rate.</summary>
        public void Append(float[] pcm, int sampleRate)
        {
            if (pcm == null || pcm.Length == 0)
                return;
            if (sampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate));

            lock (_lock)
            {
                // The generator may already have finished (or failed) while
                // the producer is still delivering; late audio is dropped.
                if (_disposed)
                    return;
                if (_complete)
                    throw new InvalidOperationException("Audio appended after Complete().");
                if (_sourceRate == 0)
                    _sourceRate = sampleRate;
                else if (_sourceRate != sampleRate)
                    throw new ArgumentException($"Chunk sample rate {sampleRate} differs from the first chunk's {_sourceRate}.");

                int needed = _sourceCount + pcm.Length;
                if (needed > _sourcePcm.Length)
                {
                    int newSize = Math.Max(needed, _sourcePcm.Length * 2);
                    Array.Resize(ref _sourcePcm, newSize);
                }
                Array.Copy(pcm, 0, _sourcePcm, _sourceCount, pcm.Length);
                _sourceCount = needed;
            }
        }

        /// <summary>The utterance has ended; the next <see cref="UpdateAsync"/> emits every remaining frame.</summary>
        public void Complete()
        {
            lock (_lock)
                _complete = true;
        }

        /// <summary>The producer failed; the generator stops at its next check and rethrows.</summary>
        public void Fail(Exception error)
        {
            lock (_lock)
                _error ??= error ?? new InvalidOperationException("Streaming audio failed.");
        }

        /// <summary>Audio has arrived (or completion was signalled) since the last <see cref="UpdateAsync"/>.</summary>
        public bool HasPendingInput
        {
            get
            {
                lock (_lock)
                    return _sourceCount > _consumedSourceCount || (_complete && !_completeConsumed) || _error != null;
            }
        }

        private bool _completeConsumed;

        public bool IsComplete { get { lock (_lock) return _complete; } }

        public Exception Error { get { lock (_lock) return _error; } }

        /// <summary>Seconds of source audio appended so far.</summary>
        public float AccumulatedSeconds
        {
            get
            {
                lock (_lock)
                    return _sourceRate > 0 ? (float)_sourceCount / _sourceRate : 0f;
            }
        }

        #endregion

        #region Compute side

        /// <summary>
        /// Output frames whose features are final under the local dependency
        /// analysis (see the class remarks). Only advances; equals
        /// <see cref="TotalFrameCount"/> after the completing update.
        /// </summary>
        public int SafeFrameCount => _safeFrameCount;

        /// <summary>Frame count of the finished utterance, or -1 until <see cref="Complete"/> has been processed.</summary>
        public int TotalFrameCount => _totalFrameCount;

        /// <summary>Whether the completing update has run: every frame is now safe.</summary>
        public bool IsFinal => _totalFrameCount >= 0 && _safeFrameCount >= _totalFrameCount;

        /// <summary><see cref="UpdateAsync"/> calls that consumed input so far (at most one encoder run each).</summary>
        public int UpdateCount { get; private set; }

        /// <summary>
        /// Audio held back (in seconds) before a frame counts as safe while the
        /// utterance is still arriving. Derived from the constants, not chosen:
        /// frame <c>i</c> needs samples up to
        /// <c>LastMelColumnFor(i) * HOP + N_FFT / 2</c>, and its nominal time
        /// is <c>i / FPS</c>.
        /// </summary>
        public static float MarginSeconds =>
            (float)AudioUtils.MelColumnEndSample(WhisperModel.LastMelColumnFor(0)) / AudioUtils.SAMPLE_RATE;

        /// <summary>Warm-up in effect for this instance.</summary>
        public float WarmupSeconds => _warmupSeconds;

        /// <summary>Extra hold-back in effect for this instance (see the constructor).</summary>
        public float ExtraContextSeconds => _extraContextSeconds;

        public Task StartSessionAsync() 
        {
            _sessionStarted = true;
            return _whisper.StartSession();
        }

        public void EndSession()
        {
            if (!_sessionStarted)
                return;
            _sessionStarted = false;
            _whisper.EndSession();
        }

        /// <summary>
        /// Recomputes the features over everything appended so far: new mel
        /// columns, the (frozen) log scale, one encoder run, and the new
        /// <see cref="SafeFrameCount"/>. Returns false when nothing could be
        /// done yet (less than the warm-up and not complete). Heavy work runs
        /// on a worker; the encoder call is the model's own async run.
        /// </summary>
        public async Task<bool> UpdateAsync()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(StreamingAudioFeatures));
            try
            {
                return await UpdateCoreAsync();
            }
            finally
            {
                UpdateCount++;
            }
        }

        private async Task<bool> UpdateCoreAsync()
        {
            float[] source;
            int sourceCount, sourceRate;
            bool complete;
            Exception error;
            lock (_lock)
            {
                error = _error;
                sourceCount = _sourceCount;
                sourceRate = _sourceRate;
                complete = _complete;
                source = new float[sourceCount];
                Array.Copy(_sourcePcm, source, sourceCount);
                _consumedSourceCount = sourceCount;
                if (complete)
                    _completeConsumed = true;
            }

            if (error != null)
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error).Throw();

            if (sourceCount == 0)
            {
                if (complete)
                {
                    // An empty utterance: nothing to animate.
                    _totalFrameCount = 0;
                    _safeFrameCount = 0;
                    return true;
                }
                return false;
            }

            // Resample the whole prefix, as the batch path resamples the whole
            // clip. Linear interpolation is local, so every output sample but
            // the last matches the finished clip's; the last is inside the
            // margin. No-op at 16 kHz, which is what an animatable character
            // requests.
            float[] audio16 = sourceRate == AudioUtils.SAMPLE_RATE
                ? source
                : AudioUtils.ResampleAudio(source, sourceRate, AudioUtils.SAMPLE_RATE);
            int length16k = audio16.Length;

            float warmupSamples = _warmupSeconds * AudioUtils.SAMPLE_RATE;
            if (!_referenceFrozen && !complete && length16k < warmupSamples)
                return false;

            float[,] mel = null;
            int numFrames = 0;
            await Task.Run(() =>
            {
                numFrames = AudioUtils.MelFrameCount(length16k);
                float[] padded = AudioUtils.BuildPaddedAudio(audio16);
                _windowScratch ??= new float[AudioUtils.N_FFT];

                mel = new float[AudioUtils.N_MELS, numFrames];
                var column = new float[AudioUtils.N_MELS];
                for (int frame = 0; frame < numFrames; frame++)
                {
                    float[] values;
                    if (frame < _finalColumns.Count)
                    {
                        values = _finalColumns[frame];
                    }
                    else
                    {
                        AudioUtils.ComputeMelPowerColumn(padded, frame, column, _windowScratch);
                        values = column;
                        // Final once the window cannot reach the end padding.
                        // Columns are processed in order, so the cache stays
                        // a contiguous prefix.
                        if (frame == _finalColumns.Count && AudioUtils.MelColumnEndSample(frame) <= length16k)
                            _finalColumns.Add((float[])column.Clone());
                    }
                    for (int m = 0; m < AudioUtils.N_MELS; m++)
                        mel[m, frame] = values[m];
                }

                if (!_referenceFrozen)
                {
                    _reference = AudioUtils.MaxMelPower(mel, numFrames);
                    _referenceFrozen = true;
                    Logger.LogVerbose($"[StreamingAudioFeatures] Reference captured at {length16k / (float)AudioUtils.SAMPLE_RATE:F2}s of audio (max mel power {_reference:E3})");
                }
                else if (_raiseReference)
                {
                    float max = AudioUtils.MaxMelPower(mel, numFrames);
                    if (max > _reference)
                    {
                        Logger.LogVerbose($"[StreamingAudioFeatures] Reference raised {10f * Mathf.Log10(max / _reference):F2} dB at {length16k / (float)AudioUtils.SAMPLE_RATE:F2}s");
                        _reference = max;
                    }
                }
                AudioUtils.ConvertToLogScale(mel, numFrames, _reference);
            });

            _encoding = await _whisper.EncodeAsync(mel);
            _encodedLength16k = length16k;
            _actualTimesteps = WhisperModel.ActualTimestepsFor(length16k, _encoding.SeqLen);

            int frameCount = WhisperModel.ChunkableFrameCount(WhisperModel.FrameCountFor(length16k), _actualTimesteps);
            if (complete)
            {
                _totalFrameCount = frameCount;
                _safeFrameCount = frameCount;
            }
            else
            {
                // Largest i with MelColumnEndSample(LastMelColumnFor(i)) + extra <= length16k.
                int extra = Mathf.RoundToInt(_extraContextSeconds * AudioUtils.SAMPLE_RATE);
                int safe = 0;
                while (safe < frameCount
                       && AudioUtils.MelColumnEndSample(WhisperModel.LastMelColumnFor(safe)) + extra <= length16k)
                    safe++;
                // Never fewer than already promised (length only grows).
                _safeFrameCount = Mathf.Max(_safeFrameCount, safe);
            }
            return true;
        }

        /// <summary>
        /// Feature chunk for <paramref name="frameIndex"/> from the most recent
        /// encoder run. Valid for <c>frameIndex &lt; SafeFrameCount</c>.
        /// </summary>
        public float[] GetFrameChunk(int frameIndex)
        {
            if (_encoding == null)
                throw new InvalidOperationException("No features yet; call UpdateAsync first.");
            if (frameIndex < 0 || frameIndex >= _safeFrameCount)
                throw new ArgumentOutOfRangeException(nameof(frameIndex),
                    $"Frame {frameIndex} is not safe yet (safe count {_safeFrameCount}, audio {_encodedLength16k / (float)AudioUtils.SAMPLE_RATE:F2}s).");
            return WhisperModel.BuildFrameChunk(_encoding, _actualTimesteps, frameIndex);
        }

        #endregion

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed)
                    return;
                _disposed = true;
                _sourcePcm = Array.Empty<float>();
                _sourceCount = 0;
            }
            EndSession();
            _finalColumns.Clear();
            _encoding = null;
        }
    }
}
