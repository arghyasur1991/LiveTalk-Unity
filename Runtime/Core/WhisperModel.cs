using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace LiveTalk.Core
{
    using API;
    using Utils;
    
    /// <summary>
    /// ONNX Whisper model implementation for audio feature extraction and processing.
    /// This class provides comprehensive audio processing capabilities including mel spectrogram extraction,
    /// Whisper model inference, and feature chunk generation for MuseTalk synchronization.
    /// </summary>
    internal class WhisperModel : IDisposable
    {
        #region Private Fields
        private Model _model;
        private bool _isInitialized = false;
        private bool _disposed = false;
        private const string OUTPUT_NAME = "audio_features_all_layers"; // ONNX output tensor name

        #endregion

        #region Feature layout constants

        /// <summary>Output (video) frame rate the feature chunks are cut for.</summary>
        internal const int FPS = 25;

        /// <summary>Whisper encoder timestep rate: one timestep per two mel columns.</summary>
        internal const int AUDIO_FPS = 50;

        /// <summary>Timesteps of context before a frame's own two timesteps (in units of <c>AUDIO_FPS / FPS</c>).</summary>
        internal const int AUDIO_PADDING_LEFT = 2;

        /// <summary>Timesteps of lookahead after a frame's own two timesteps (same units).</summary>
        internal const int AUDIO_PADDING_RIGHT = 2;

        /// <summary>Mel columns per encoder timestep (the stride-2 conv in the Whisper front end).</summary>
        internal const int MEL_FRAMES_PER_TIMESTEP = 2;

        /// <summary>
        /// Extra mel columns an encoder timestep reads past its own two: the
        /// Whisper front end is two kernel-3 convolutions, each reaching one
        /// column further.
        /// </summary>
        internal const int CONV_REACH_MEL_FRAMES = 2;

        /// <summary><c>AUDIO_FPS / FPS</c>: encoder timesteps per output frame.</summary>
        internal static int TimestepsPerFrame => Mathf.CeilToInt((float)AUDIO_FPS / FPS);

        /// <summary>Encoder timesteps each frame's chunk holds: <c>2 * (left + right + 1)</c> = 10.</summary>
        internal static int TimestepsPerChunk => TimestepsPerFrame * (AUDIO_PADDING_LEFT + AUDIO_PADDING_RIGHT + 1);

        /// <summary>Zero timesteps prepended before the first real one.</summary>
        internal static int LeftPaddingTimesteps => TimestepsPerFrame * AUDIO_PADDING_LEFT;

        /// <summary>Number of output frames for <paramref name="audioLength16k"/> samples at 16 kHz.</summary>
        internal static int FrameCountFor(int audioLength16k) =>
            Mathf.FloorToInt((float)audioLength16k / AudioUtils.SAMPLE_RATE * FPS);

        /// <summary>Encoder timesteps that carry real audio for <paramref name="audioLength16k"/> samples, capped by the encoder's sequence length.</summary>
        internal static int ActualTimestepsFor(int audioLength16k, int seqLen) =>
            Mathf.Min(Mathf.FloorToInt((float)audioLength16k / AudioUtils.SAMPLE_RATE * AUDIO_FPS), seqLen);

        /// <summary>
        /// Frames whose chunk fits inside the padded sequence (left zeros + real
        /// timesteps + right zeros). Equal to <paramref name="numFrames"/> unless
        /// the clip is longer than the encoder's 30 s window, in which case the
        /// tail is dropped — the long-standing batch behaviour.
        /// </summary>
        internal static int ChunkableFrameCount(int numFrames, int actualTimesteps)
        {
            int rightPaddingSize = TimestepsPerFrame * 3 * AUDIO_PADDING_RIGHT;
            int totalPaddedLength = LeftPaddingTimesteps + actualTimesteps + rightPaddingSize;
            int count = 0;
            while (count < numFrames && TimestepsPerFrame * count + TimestepsPerChunk <= totalPaddedLength)
                count++;
            return count;
        }

        /// <summary>
        /// The last mel column output frame <paramref name="frameIndex"/> depends on
        /// through the local part of the pipeline (chunk window, conv front end,
        /// STFT window). Global self-attention in the encoder is not counted —
        /// it reaches everything and cannot be held back.
        /// </summary>
        internal static int LastMelColumnFor(int frameIndex)
        {
            int lastTimestep = TimestepsPerFrame * frameIndex + TimestepsPerChunk - 1 - LeftPaddingTimesteps;
            return MEL_FRAMES_PER_TIMESTEP * lastTimestep + (MEL_FRAMES_PER_TIMESTEP - 1) + CONV_REACH_MEL_FRAMES;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets a value indicating whether the Whisper model is initialized and ready for inference.
        /// </summary>
        public bool IsInitialized => _isInitialized;

        /// <summary>
        /// Gets the loading task for the underlying model.
        /// </summary>
        public Task LoadTask => _model?.LoadTask;

        #endregion

        #region Public Methods - Loading

        /// <summary>
        /// Waits for the model to be fully loaded.
        /// </summary>
        public async Task WaitForLoadAsync()
        {
            if (_model?.LoadTask != null)
            {
                await _model.LoadTask;
            }
        }

        #endregion

        #region Constructor
        
        /// <summary>
        /// Initializes a new instance of the WhisperModel class with the specified configuration.
        /// Loads the Whisper encoder ONNX model from StreamingAssets for audio feature extraction.
        /// </summary>
        /// <param name="config">The LiveTalk configuration containing model paths and settings</param>
        /// <exception cref="ArgumentNullException">Thrown when config is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when model loading fails</exception>
        public WhisperModel(LiveTalkConfig config)
        {
            _model = new Model(config, "whisper_encoder", "MuseTalk", ExecutionProvider.CPU);
            _isInitialized = true;
        }

        #endregion

        #region Public Methods
        
        /// <summary>
        /// Asynchronously processes audio samples and extracts Whisper features using pure Unity/C# implementation.
        /// This method performs the complete audio processing pipeline including resampling, mel spectrogram extraction,
        /// Whisper inference, and feature chunk generation.
        /// </summary>
        /// <param name="audioSamples">The input audio samples to process</param>
        /// <param name="originalSampleRate">The original sample rate of the audio data (default: 44100 Hz)</param>
        /// <returns>A task containing the extracted audio features formatted for MuseTalk synchronization</returns>
        /// <exception cref="InvalidOperationException">Thrown when the model is not initialized</exception>
        /// <exception cref="ArgumentNullException">Thrown when audioSamples is null or empty</exception>
        public async Task<AudioFeatures> ProcessAudio(float[] audioSamples, int originalSampleRate = 44100)
        {
            if (!_isInitialized)
            {
                Logger.LogError("[WhisperModel] Model not initialized");
                return null;
            }
            
            if (audioSamples == null || audioSamples.Length == 0)
            {
                Logger.LogError("[WhisperModel] Audio samples are null or empty");
                return null;
            }
            
            try
            {
                // Step 1: Resample to 16kHz if needed
                float[] resampledAudio = audioSamples;
                if (originalSampleRate != AudioUtils.SAMPLE_RATE)
                {
                    resampledAudio = AudioUtils.ResampleAudio(audioSamples, originalSampleRate, AudioUtils.SAMPLE_RATE);
                }
                
                // Step 2: Extract mel spectrogram
                float[,] melSpectrogram = AudioUtils.ExtractMelSpectrogram(resampledAudio);
                
                // Step 3: Process through ONNX Whisper
                var encoding = await EncodeAsync(melSpectrogram);
                
                // Step 4: Convert to MuseTalk audio chunks
                var audioFeatures = ProcessWhisperFeatures(encoding, resampledAudio.Length);
                
                return audioFeatures;
            }
            catch (Exception e)
            {
                Logger.LogError($"[WhisperModel] Error processing audio: {e.Message}");
                return null;
            }
        }

        /// <summary>
        /// Starts the session for the Whisper model
        /// </summary>
        public async Task StartSession()
        {
            await _model.StartSession();
        }

        /// <summary>
        /// Ends the session for the Whisper model
        /// </summary>
        public void EndSession()
        {
            _model.EndSession();
        }

        #endregion

        #region Encoder output

        /// <summary>
        /// One run of the Whisper encoder: <c>[seqLen, layers, features]</c>
        /// flattened row-major (<c>s * layers * features + l * features + f</c>).
        /// </summary>
        internal sealed class Encoding
        {
            public float[] Data;
            public int SeqLen;
            public int Layers;
            public int Features;

            public int ChunkLength => TimestepsPerChunk * Layers * Features;
        }

        /// <summary>
        /// Runs the encoder over a mel spectrogram, zero-padded to
        /// <see cref="AudioUtils.TARGET_FRAMES"/> columns as the model requires.
        /// The session must be started.
        /// </summary>
        internal async Task<Encoding> EncodeAsync(float[,] melSpectrogram)
        {
            int melBands = melSpectrogram.GetLength(0);
            int frames = melSpectrogram.GetLength(1);
            
            // Pad to target frames. Row-major [1, melBands, TARGET_FRAMES].
            var flat = new float[melBands * AudioUtils.TARGET_FRAMES];
            int copyFrames = Mathf.Min(frames, AudioUtils.TARGET_FRAMES);
            for (int mel = 0; mel < melBands; mel++)
            {
                int row = mel * AudioUtils.TARGET_FRAMES;
                for (int frame = 0; frame < copyFrames; frame++)
                    flat[row + frame] = melSpectrogram[mel, frame];
                // else: padding with zeros (default initialization)
            }
            
            var inputShape = new int[] { 1, melBands, AudioUtils.TARGET_FRAMES };
            var tensor = new DenseTensor<float>(flat, inputShape);
            var inputs = new List<Tensor<float>> { tensor };
            
            var outputs = await _model.Run(inputs);
            var output = outputs.First(o => o.Name == OUTPUT_NAME);
            
            if (output.Value is DenseTensor<float> outputTensor)
            {
                // Expected shape: [batch, seq_len, layers, features] = [1, seq_len, layers, 384]
                var shape = outputTensor.Dimensions.ToArray();
                if (shape.Length != 4 || shape[0] != 1)
                    throw new InvalidOperationException(
                        $"Unexpected Whisper output shape [{string.Join(", ", shape)}]; expected [1, seq, layers, features]");

                // The preallocated output buffer is reused by the next run, so
                // the caller gets its own copy.
                return new Encoding
                {
                    Data = outputTensor.Buffer.ToArray(),
                    SeqLen = shape[1],
                    Layers = shape[2],
                    Features = shape[3],
                };
            }
            
            throw new InvalidOperationException("Failed to get valid output tensor from Whisper ONNX");
        }

        /// <summary>
        /// The feature chunk for output frame <paramref name="frameIndex"/>:
        /// <see cref="TimestepsPerChunk"/> consecutive encoder timesteps starting
        /// <see cref="LeftPaddingTimesteps"/> before <c>2 * frameIndex</c>, with
        /// zeros where a timestep falls before the audio or at or after
        /// <paramref name="actualTimesteps"/>. Layout
        /// <c>(t * layers + l) * features + f</c>, which is what
        /// <c>MuseTalkInference.PrepareAudioBatch</c> reads.
        /// </summary>
        internal static float[] BuildFrameChunk(Encoding encoding, int actualTimesteps, int frameIndex)
        {
            int layers = encoding.Layers;
            int features = encoding.Features;
            int perTimestep = layers * features;
            var chunk = new float[TimestepsPerChunk * perTimestep];
            int firstPadded = TimestepsPerFrame * frameIndex;
            for (int t = 0; t < TimestepsPerChunk; t++)
            {
                int s = firstPadded + t - LeftPaddingTimesteps;
                if (s < 0 || s >= actualTimesteps || s >= encoding.SeqLen)
                    continue; // zero padding
                Array.Copy(encoding.Data, s * perTimestep, chunk, t * perTimestep, perTimestep);
            }
            return chunk;
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Processes Whisper features into MuseTalk audio chunks for temporal synchronization.
        /// This method converts the multi-layer Whisper features into frame-based chunks with proper
        /// padding and temporal alignment.
        /// </summary>
        /// <param name="encoding">The encoder output for the whole clip</param>
        /// <param name="audioLength">The original audio length in samples for duration calculation</param>
        /// <returns>AudioFeatures object containing properly formatted feature chunks for MuseTalk</returns>
        private AudioFeatures ProcessWhisperFeatures(Encoding encoding, int audioLength)
        {
            int numFrames = FrameCountFor(audioLength);
            int actualLength = ActualTimestepsFor(audioLength, encoding.SeqLen);
            
            // BuildFrameChunk zero-fills past actualLength exactly as the
            // padded array did.
            numFrames = ChunkableFrameCount(numFrames, actualLength);
            var featureChunks = new List<float[]>(numFrames);
            for (int frameIndex = 0; frameIndex < numFrames; frameIndex++)
            {
                featureChunks.Add(BuildFrameChunk(encoding, actualLength, frameIndex));
            }
            
            var audioFeatures = new AudioFeatures
            {
                FeatureChunks = featureChunks,
                SampleRate = AudioUtils.SAMPLE_RATE,
                Duration = (float)audioLength / AudioUtils.SAMPLE_RATE
            };
            return audioFeatures;
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Releases all resources used by the WhisperModel instance.
        /// Disposes of the ONNX model and resets initialization state.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases the unmanaged resources used by the WhisperModel and optionally releases the managed resources.
        /// </summary>
        /// <param name="disposing">True to release both managed and unmanaged resources; false to release only unmanaged resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources (ONNX model)
                    _model?.Dispose();
                    _model = null;
                    
                    Logger.LogVerbose("[WhisperModel] Model disposed successfully");
                }
                
                // Reset state
                _isInitialized = false;
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for the WhisperModel class.
        /// </summary>
        ~WhisperModel()
        {
            Dispose(false);
        }

        #endregion
    }
}
