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

        #region Properties

        /// <summary>
        /// Gets a value indicating whether the Whisper model is initialized and ready for inference.
        /// </summary>
        public bool IsInitialized => _isInitialized;

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
                var whisperFeatures = await RunWhisperInference(melSpectrogram);
                
                // Step 4: Convert to MuseTalk audio chunks
                var audioFeatures = ProcessWhisperFeatures(whisperFeatures, resampledAudio.Length);
                
                return audioFeatures;
            }
            catch (Exception e)
            {
                Logger.LogError($"[WhisperModel] Error processing audio: {e.Message}");
                return null;
            }
        }

        #endregion

        #region Private Methods
        
        /// <summary>
        /// Asynchronously runs ONNX Whisper model inference on the mel spectrogram.
        /// This method prepares the input tensor, executes the Whisper encoder model,
        /// and returns the multi-layer audio features for temporal analysis.
        /// </summary>
        /// <param name="melSpectrogram">The mel spectrogram to process [mel_bands, time_frames]</param>
        /// <returns>A task containing the Whisper features [batch, sequence_length, layers, features]</returns>
        /// <exception cref="InvalidOperationException">Thrown when ONNX inference fails</exception>
        private async Task<float[,,,]> RunWhisperInference(float[,] melSpectrogram)
        {
            int melBands = melSpectrogram.GetLength(0);
            int frames = melSpectrogram.GetLength(1);
            
            // Pad to target frames
            float[,,] inputTensor = new float[1, melBands, AudioUtils.TARGET_FRAMES];
            
            for (int mel = 0; mel < melBands; mel++)
            {
                for (int frame = 0; frame < AudioUtils.TARGET_FRAMES; frame++)
                {
                    if (frame < frames)
                    {
                        inputTensor[0, mel, frame] = melSpectrogram[mel, frame];
                    }
                    // else: padding with zeros (default initialization)
                }
            }
            
            // Create ONNX tensor
            var inputShape = new int[] { 1, melBands, AudioUtils.TARGET_FRAMES };
            var tensor = new DenseTensor<float>(inputTensor.Cast<float>().ToArray(), inputShape);
            var inputs = new List<Tensor<float>>
            {
                tensor
            };
            
            var outputs = await _model.Run(inputs);
            var output = outputs.First(o => o.Name == OUTPUT_NAME);
            
            if (output.Value is DenseTensor<float> outputTensor)
            {
                // Expected shape: [batch, seq_len, layers, features] = [1, seq_len, layers, 384]
                var shape = outputTensor.Dimensions.ToArray();
                
                // Convert to 4D array for easier processing
                int batchSize = (int)shape[0];
                int seqLen = (int)shape[1];
                int layers = (int)shape[2];
                int features = (int)shape[3];
                
                float[,,,] result = new float[batchSize, seqLen, layers, features];
                var buffer = outputTensor.Buffer.ToArray();
                
                for (int b = 0; b < batchSize; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        for (int l = 0; l < layers; l++)
                        {
                            for (int f = 0; f < features; f++)
                            {
                                int index = b * seqLen * layers * features + 
                                           s * layers * features + 
                                           l * features + f;
                                result[b, s, l, f] = buffer[index];
                            }
                        }
                    }
                }
                
                return result;
            }
            
            throw new InvalidOperationException("Failed to get valid output tensor from Whisper ONNX");
        }

        /// <summary>
        /// Processes Whisper features into MuseTalk audio chunks for temporal synchronization.
        /// This method converts the multi-layer Whisper features into frame-based chunks with proper
        /// padding and temporal alignment.
        /// </summary>
        /// <param name="whisperFeatures">The raw Whisper features [batch, sequence_length, layers, features]</param>
        /// <param name="audioLength">The original audio length in samples for duration calculation</param>
        /// <returns>AudioFeatures object containing properly formatted feature chunks for MuseTalk</returns>
        private AudioFeatures ProcessWhisperFeatures(float[,,,] whisperFeatures, int audioLength)
        {
            const int fps = 25;
            const int audioFps = 50;
            const int audioPaddingLeft = 2;
            const int audioPaddingRight = 2;
            
            // Get dimensions [batch, seq_len, layers, features]
            int seqLen = whisperFeatures.GetLength(1);
            int layers = whisperFeatures.GetLength(2);
            int features = whisperFeatures.GetLength(3);
            
            // Calculate parameters
            float whisperIdxMultiplier = (float)audioFps / fps;
            int numFrames = Mathf.FloorToInt((float)audioLength / AudioUtils.SAMPLE_RATE * fps);
            int actualLength = Mathf.FloorToInt((float)audioLength / AudioUtils.SAMPLE_RATE * audioFps);
            
            // Trim to actual length
            actualLength = Mathf.Min(actualLength, seqLen);
            
            // Add padding
            int paddingNums = Mathf.CeilToInt(whisperIdxMultiplier);
            int leftPaddingSize = paddingNums * audioPaddingLeft;
            int rightPaddingSize = paddingNums * 3 * audioPaddingRight;
            
            int totalPaddedLength = leftPaddingSize + actualLength + rightPaddingSize;
            
            // Create padded features array
            float[,,,] paddedFeatures = new float[1, totalPaddedLength, layers, features];
            
            // Copy actual features to padded array (padding is zeros by default)
            for (int s = 0; s < actualLength; s++)
            {
                for (int l = 0; l < layers; l++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        paddedFeatures[0, leftPaddingSize + s, l, f] = whisperFeatures[0, s, l, f];
                    }
                }
            }
            
            // Generate chunks
            int audioFeatureLengthPerFrame = 2 * (audioPaddingLeft + audioPaddingRight + 1);
            var featureChunks = new List<float[]>();
            
            for (int frameIndex = 0; frameIndex < numFrames; frameIndex++)
            {
                int audioIndex = Mathf.FloorToInt(frameIndex * whisperIdxMultiplier);
                
                if (audioIndex + audioFeatureLengthPerFrame <= totalPaddedLength)
                {
                    int chunkHeight = audioFeatureLengthPerFrame * layers;
                    float[] chunk = new float[chunkHeight * features];
                    
                    for (int t = 0; t < audioFeatureLengthPerFrame; t++)
                    {
                        for (int l = 0; l < layers; l++)
                        {
                            for (int f = 0; f < features; f++)
                            {
                                int rowIndex = t * layers + l;
                                int chunkIndex = rowIndex * features + f;
                                chunk[chunkIndex] = paddedFeatures[0, audioIndex + t, l, f];
                            }
                        }
                    }
                    
                    featureChunks.Add(chunk);
                }
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
