using System;
using UnityEngine;

namespace LiveTalk.API
{
    /// <summary>
    /// A line of speech while it is still being made: the PCM synthesised so
    /// far plus the <see cref="FrameStream"/> its lip-sync frames arrive on.
    /// Handed to <c>Character.SpeakAsync</c>'s <c>onStreamStarted</c> as soon
    /// as the first audio chunk exists, which is seconds before
    /// <c>onAudioReady</c> delivers the finished <see cref="AudioClip"/>.
    ///
    /// <para>Audio is append-only and read by absolute sample index, so a
    /// consumer can drive a streaming <see cref="AudioClip"/>'s PCM reader
    /// from the audio thread (<see cref="ReadSamples"/> is thread-safe and
    /// never blocks; samples not yet synthesised read as silence and are
    /// reported). Frame <c>i</c> belongs at audio time <c>i / 25</c> s.</para>
    /// </summary>
    public sealed class SpeechStream
    {
        private readonly object _lock = new();
        private float[] _pcm;
        private int _count;
        private bool _finished;
        private Exception _error;

        internal SpeechStream(FrameStream frames, int sampleRate)
        {
            Frames = frames ?? throw new ArgumentNullException(nameof(frames));
            if (sampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate));
            SampleRate = sampleRate;
            _pcm = new float[sampleRate * 8];
        }

        /// <summary>The lip-sync frames for this line, in order, at 25 fps. Same object <c>onAudioReady</c> later receives.</summary>
        public FrameStream Frames { get; }

        /// <summary>Sample rate of the PCM. Mono.</summary>
        public int SampleRate { get; }

        /// <summary>Samples synthesised so far.</summary>
        public int SamplesAvailable
        {
            get { lock (_lock) return _count; }
        }

        /// <summary><see cref="SamplesAvailable"/> in seconds.</summary>
        public float SecondsAvailable => (float)SamplesAvailable / SampleRate;

        /// <summary>True once the last chunk has been appended; <see cref="SamplesAvailable"/> is then the clip length.</summary>
        public bool AudioFinished
        {
            get { lock (_lock) return _finished; }
        }

        /// <summary>The synthesis failure, if any. The frames stream carries the same error.</summary>
        public Exception Error
        {
            get { lock (_lock) return _error; }
        }

        /// <summary>
        /// Copies samples <c>[start, start + count)</c> into <paramref name="destination"/>.
        /// Samples not yet available are written as zeros. Returns how many
        /// real samples were copied; less than <paramref name="count"/> means
        /// the reader is ahead of synthesis (an underrun) unless
        /// <see cref="AudioFinished"/>, in which case the clip simply ended.
        /// Safe from any thread.
        /// </summary>
        public int ReadSamples(int start, float[] destination, int destinationOffset, int count)
        {
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));
            if (count <= 0)
                return 0;
            if (start < 0)
                start = 0;

            int copied;
            lock (_lock)
            {
                int available = Math.Max(0, Math.Min(count, _count - start));
                if (available > 0)
                    Array.Copy(_pcm, start, destination, destinationOffset, available);
                copied = available;
            }
            if (copied < count)
                Array.Clear(destination, destinationOffset + copied, count - copied);
            return copied;
        }

        internal void Append(float[] pcm)
        {
            if (pcm == null || pcm.Length == 0)
                return;
            lock (_lock)
            {
                if (_finished)
                    throw new InvalidOperationException("Audio appended after the stream finished.");
                int needed = _count + pcm.Length;
                if (needed > _pcm.Length)
                    Array.Resize(ref _pcm, Math.Max(needed, _pcm.Length * 2));
                Array.Copy(pcm, 0, _pcm, _count, pcm.Length);
                _count = needed;
            }
        }

        internal void Finish()
        {
            lock (_lock)
                _finished = true;
        }

        internal void Fail(Exception error)
        {
            lock (_lock)
            {
                _error ??= error;
                _finished = true;
            }
        }

        /// <summary>
        /// Set by the producer so a consumer that stops mid-line can abandon the
        /// frames still being generated for it (releasing the lip-sync engine
        /// for the next line) instead of letting them run to completion unseen.
        /// The audio synthesis is not affected.
        /// </summary>
        internal Action CancelGeneration { get; set; }

        /// <summary>True once <see cref="Cancel"/> was called.</summary>
        public bool IsCancelled { get; private set; }

        /// <summary>
        /// Abandons the lip-sync frames still being generated for this line.
        /// The frames stream finishes (with a cancellation error) and nothing
        /// is cached for the line. Idempotent; safe after the line finished.
        /// </summary>
        public void Cancel()
        {
            if (IsCancelled)
                return;
            IsCancelled = true;
            CancelGeneration?.Invoke();
        }
    }
}
