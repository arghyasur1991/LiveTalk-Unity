using System;
using UnityEngine;

namespace LiveTalk.Utils
{
    /// <summary>
    /// Comprehensive audio processing utilities for LiveTalk inference pipeline.
    /// Provides Unity-compatible audio operations including format conversion, resampling,
    /// and advanced mel spectrogram extraction with librosa compatibility for cross-platform consistency.
    /// All methods are optimized for real-time audio processing in Unity environments.
    /// </summary>
    internal static class AudioUtils
    {
        #region Audio Processing Constants

        /// <summary>
        /// Target sample rate for Whisper model processing (16kHz).
        /// This matches the Whisper model's expected input format and provides optimal balance
        /// between audio quality and computational efficiency for speech recognition.
        /// </summary>
        internal static readonly int SAMPLE_RATE = 16000;

        /// <summary>
        /// Number of mel frequency bands for mel spectrogram generation (80 bands).
        /// This configuration matches librosa's default mel filterbank for speech processing
        /// and provides sufficient frequency resolution for Whisper feature extraction.
        /// </summary>
        internal static readonly int N_MELS = 80;

        /// <summary>
        /// Hop length for STFT analysis in samples (160 samples = 10ms at 16kHz).
        /// This provides 100 FPS temporal resolution for audio features, matching
        /// typical video frame rates and enabling precise audio-visual synchronization.
        /// </summary>
        internal static readonly int HOP_LENGTH = 160;

        /// <summary>
        /// Window length for STFT analysis in samples (400 samples = 25ms at 16kHz).
        /// This window size provides good frequency resolution while maintaining
        /// temporal precision for speech analysis, following librosa conventions.
        /// </summary>
        internal static readonly int WIN_LENGTH = 400;

        /// <summary>
        /// Target number of time frames for padded mel spectrograms (3000 frames = 30 seconds).
        /// This standardizes audio input length for batch processing and ensures
        /// consistent tensor dimensions for neural network inference.
        /// </summary>
        internal static readonly int TARGET_FRAMES = 3000;

        #endregion

        #region Public Methods - Audio Format Conversion

        /// <summary>
        /// Converts Unity AudioClip to float array for processing.
        /// This method extracts raw audio samples from Unity's AudioClip format into a standard
        /// float array suitable for mathematical operations and neural network preprocessing.
        /// </summary>
        /// <param name="audioClip">The Unity AudioClip to convert to float array</param>
        /// <returns>Float array containing raw audio samples with interleaved channels if stereo</returns>
        /// <exception cref="ArgumentNullException">Thrown when audioClip is null</exception>
        public static float[] AudioClipToFloatArray(AudioClip audioClip)
        {
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
                
            float[] samples = new float[audioClip.samples * audioClip.channels];
            audioClip.GetData(samples, 0);
            
            return samples;
        }
        
        /// <summary>
        /// Creates Unity AudioClip from float array for playback.
        /// This method converts processed audio samples back into Unity's AudioClip format
        /// for playback, testing, or integration with Unity's audio system.
        /// </summary>
        /// <param name="samples">The float array containing audio samples to convert</param>
        /// <param name="sampleRate">The sample rate of the audio data in Hz</param>
        /// <param name="channels">The number of audio channels (1 for mono, 2 for stereo)</param>
        /// <param name="name">The name to assign to the created AudioClip</param>
        /// <returns>A Unity AudioClip ready for playback or further processing</returns>
        /// <exception cref="ArgumentNullException">Thrown when samples is null</exception>
        public static AudioClip FloatArrayToAudioClip(float[] samples, int sampleRate, int channels = 1, string name = "GeneratedAudio")
        {
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));
                
            int sampleCount = samples.Length / channels;
            AudioClip clip = AudioClip.Create(name, sampleCount, channels, sampleRate, false);
            clip.SetData(samples, 0);
            
            return clip;
        }

        /// <summary>
        /// Converts stereo audio to mono by averaging left and right channels.
        /// This method reduces computational overhead by converting stereo audio to mono
        /// while preserving the essential audio characteristics needed for speech processing.
        /// </summary>
        /// <param name="stereoSamples">The stereo audio samples with interleaved left/right channels</param>
        /// <returns>Mono audio samples with averaged channel data</returns>
        /// <exception cref="ArgumentNullException">Thrown when stereoSamples is null</exception>
        /// <exception cref="ArgumentException">Thrown when stereo samples array length is not even</exception>
        public static float[] StereoToMono(float[] stereoSamples)
        {
            if (stereoSamples == null)
                throw new ArgumentNullException(nameof(stereoSamples));
                
            if (stereoSamples.Length % 2 != 0)
                throw new ArgumentException("Stereo samples array length must be even", nameof(stereoSamples));
                
            float[] monoSamples = new float[stereoSamples.Length / 2];
            
            for (int i = 0; i < monoSamples.Length; i++)
            {
                monoSamples[i] = (stereoSamples[i * 2] + stereoSamples[i * 2 + 1]) * 0.5f;
            }
            
            return monoSamples;
        }

        #endregion

        #region Public Methods - Audio Resampling

        /// <summary>
        /// Resamples audio to target sample rate using linear interpolation.
        /// This method changes the sample rate of audio data while preserving the audio content,
        /// using linear interpolation for smooth transitions between samples. Optimized for real-time processing.
        /// </summary>
        /// <param name="inputSamples">The input audio samples to resample</param>
        /// <param name="originalSampleRate">The current sample rate of the input audio in Hz</param>
        /// <param name="targetSampleRate">The desired output sample rate in Hz</param>
        /// <returns>Resampled audio data at the target sample rate</returns>
        /// <exception cref="ArgumentNullException">Thrown when inputSamples is null</exception>
        public static float[] ResampleAudio(float[] inputSamples, int originalSampleRate, int targetSampleRate)
        {
            if (inputSamples == null)
                throw new ArgumentNullException(nameof(inputSamples));
                
            if (originalSampleRate == targetSampleRate)
                return inputSamples;
                
            float ratio = (float)targetSampleRate / originalSampleRate;
            int outputLength = Mathf.RoundToInt(inputSamples.Length * ratio);
            float[] outputSamples = new float[outputLength];
            
            for (int i = 0; i < outputLength; i++)
            {
                float sourceIndex = i / ratio;
                int index = Mathf.FloorToInt(sourceIndex);
                float fraction = sourceIndex - index;
                
                if (index < inputSamples.Length - 1)
                {
                    // Linear interpolation between adjacent samples
                    outputSamples[i] = Mathf.Lerp(inputSamples[index], inputSamples[index + 1], fraction);
                }
                else if (index < inputSamples.Length)
                {
                    outputSamples[i] = inputSamples[index];
                }
            }
            
            return outputSamples;
        }

        #endregion

        #region Public Methods - Advanced Audio Processing

        /// <summary>
        /// Extracts mel spectrogram from audio samples using librosa-compatible implementation.
        /// This method performs STFT with Hann windowing, applies mel filterbank, and converts to log scale,
        /// matching librosa.feature.melspectrogram processing.
        /// The implementation includes proper padding, windowing, FFT computation, mel filtering, and normalization.
        /// </summary>
        /// <param name="audioSamples">The input audio samples to process (mono, 16kHz recommended)</param>
        /// <returns>A 2D array containing the mel spectrogram [mel_bands, time_frames] ready for Whisper processing</returns>
        /// <exception cref="ArgumentNullException">Thrown when audioSamples is null</exception>
        public static float[,] ExtractMelSpectrogram(float[] audioSamples)
        {
            if (audioSamples == null)
                throw new ArgumentNullException(nameof(audioSamples));

            // Use librosa-compatible parameters
            const int nFft = 512;  // Default librosa n_fft for 16kHz
            
            // Add padding for centering (matching librosa center=True)
            int halfWindow = nFft / 2;
            float[] paddedAudio = new float[audioSamples.Length + nFft - 1];
            
            // Reflect padding at the beginning with bounds checking
            for (int i = 0; i < halfWindow; i++)
            {
                int srcIndex = halfWindow - 1 - i;
                // Bounds checking: ensure srcIndex is valid
                if (srcIndex >= 0 && srcIndex < audioSamples.Length)
                {
                    paddedAudio[i] = audioSamples[srcIndex];
                }
                else
                {
                    // If audio is too short, repeat the first sample
                    paddedAudio[i] = audioSamples.Length > 0 ? audioSamples[0] : 0f;
                }
            }
            
            // Copy original audio
            Array.Copy(audioSamples, 0, paddedAudio, halfWindow, audioSamples.Length);
            
            // Reflect padding at the end with bounds checking
            for (int i = 0; i < halfWindow; i++)
            {
                int srcIndex = audioSamples.Length - 1 - i;
                int destIndex = halfWindow + audioSamples.Length + i;
                
                // Bounds checking: ensure indices are valid
                if (srcIndex >= 0 && srcIndex < audioSamples.Length && destIndex < paddedAudio.Length)
                {
                    paddedAudio[destIndex] = audioSamples[srcIndex];
                }
                else
                {
                    // If audio is too short, repeat the last sample
                    float lastSample = audioSamples.Length > 0 ? audioSamples[audioSamples.Length - 1] : 0f;
                    if (destIndex < paddedAudio.Length)
                    {
                        paddedAudio[destIndex] = lastSample;
                    }
                }
            }
            
            // Calculate number of frames (matching librosa with centering)
            int numFrames = (paddedAudio.Length - nFft) / HOP_LENGTH + 1;
            numFrames = Mathf.Min(numFrames, TARGET_FRAMES);
            
            // Create mel filterbank (matching librosa)
            float[,] melFilterBank = CreateLibrosaMelFilterBank(nFft);
            
            // Compute power spectrogram using STFT
            float[,] powerSpec = ComputePowerSpectrogram(paddedAudio, nFft, numFrames);
            
            // Apply mel filterbank to get mel spectrogram
            float[,] melSpec = ApplyMelFilterBank(powerSpec, melFilterBank, numFrames);
            
            // Convert to log scale and normalize (matching librosa.power_to_db)
            melSpec = ConvertToLogScale(melSpec, numFrames);
            
            // Pad to target frames if needed
            if (numFrames < TARGET_FRAMES)
            {
                return PadMelSpectrogram(melSpec, numFrames);
            }
            
            return melSpec;
        }

        #endregion

        #region Private Methods - STFT and Spectrogram Processing

        /// <summary>
        /// Computes power spectrogram from padded audio using Short-Time Fourier Transform.
        /// This method applies Hann windowing and computes magnitude squared FFT for each frame,
        /// following librosa's STFT implementation for consistent frequency analysis.
        /// </summary>
        /// <param name="paddedAudio">The zero-padded audio samples ready for STFT processing</param>
        /// <param name="nFft">The FFT size for frequency analysis</param>
        /// <param name="numFrames">The number of time frames to process</param>
        /// <returns>A 2D power spectrogram array [frequency_bins, time_frames]</returns>
        private static float[,] ComputePowerSpectrogram(float[] paddedAudio, int nFft, int numFrames)
        {
            float[,] powerSpec = new float[nFft / 2 + 1, numFrames];
            
            for (int frame = 0; frame < numFrames; frame++)
            {
                int startSample = frame * HOP_LENGTH;
                
                // Apply Hann window (matching librosa)
                float[] windowedFrame = new float[nFft];
                for (int i = 0; i < nFft && startSample + i < paddedAudio.Length; i++)
                {
                    float hannWindow = 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * i / (nFft - 1)));
                    windowedFrame[i] = paddedAudio[startSample + i] * hannWindow;
                }
                
                // Compute FFT magnitude squared (power) using direct DFT implementation
                for (int bin = 0; bin < nFft / 2 + 1; bin++)
                {
                    float real = 0f, imag = 0f;
                    
                    for (int i = 0; i < nFft; i++)
                    {
                        float angle = -2f * Mathf.PI * bin * i / nFft;
                        real += windowedFrame[i] * Mathf.Cos(angle);
                        imag += windowedFrame[i] * Mathf.Sin(angle);
                    }
                    
                    powerSpec[bin, frame] = real * real + imag * imag;
                }
            }
            
            return powerSpec;
        }

        /// <summary>
        /// Applies mel filterbank to power spectrogram to create mel-scaled representation.
        /// This method transforms linear frequency bins to perceptually-meaningful mel scale
        /// by applying triangular mel filters to the power spectrogram.
        /// </summary>
        /// <param name="powerSpec">The input power spectrogram [frequency_bins, time_frames]</param>
        /// <param name="melFilterBank">The mel filterbank coefficients [mel_bands, frequency_bins]</param>
        /// <param name="numFrames">The number of time frames to process</param>
        /// <returns>A mel spectrogram array [mel_bands, time_frames]</returns>
        private static float[,] ApplyMelFilterBank(float[,] powerSpec, float[,] melFilterBank, int numFrames)
        {
            int nFreqBins = powerSpec.GetLength(0);
            float[,] melSpec = new float[N_MELS, numFrames];
            
            for (int mel = 0; mel < N_MELS; mel++)
            {
                for (int frame = 0; frame < numFrames; frame++)
                {
                    float melValue = 0f;
                    
                    for (int bin = 0; bin < nFreqBins; bin++)
                    {
                        melValue += melFilterBank[mel, bin] * powerSpec[bin, frame];
                    }
                    
                    melSpec[mel, frame] = melValue;
                }
            }
            
            return melSpec;
        }

        /// <summary>
        /// Converts mel spectrogram to log scale and applies normalization.
        /// This method applies librosa's power_to_db conversion with top_db clamping
        /// and final normalization to the range [-1, 1] for neural network compatibility.
        /// </summary>
        /// <param name="melSpec">The input mel spectrogram to convert</param>
        /// <param name="numFrames">The number of time frames in the spectrogram</param>
        /// <returns>A log-scaled and normalized mel spectrogram</returns>
        private static float[,] ConvertToLogScale(float[,] melSpec, int numFrames)
        {
            // Find maximum value for reference-based normalization
            float maxValue = float.MinValue;
            for (int mel = 0; mel < N_MELS; mel++)
            {
                for (int frame = 0; frame < numFrames; frame++)
                {
                    maxValue = Mathf.Max(maxValue, melSpec[mel, frame]);
                }
            }
            
            // Convert to log scale (matching librosa.power_to_db)
            for (int mel = 0; mel < N_MELS; mel++)
            {
                for (int frame = 0; frame < numFrames; frame++)
                {
                    // power_to_db: 10 * log10(max(S, ref)) where ref = max(S)
                    float dbValue = 10f * Mathf.Log10(Mathf.Max(melSpec[mel, frame], maxValue * 1e-10f)) 
                                  - 10f * Mathf.Log10(maxValue);
                    
                    // Apply top_db=80.0 clamping (matching librosa exactly)
                    melSpec[mel, frame] = Mathf.Max(dbValue, -80f);
                }
            }
            
            // Normalize: (mel + 80.0) / 80.0, clipped to [-1, 1]
            for (int mel = 0; mel < N_MELS; mel++)
            {
                for (int frame = 0; frame < numFrames; frame++)
                {
                    melSpec[mel, frame] = (melSpec[mel, frame] + 80f) / 80f;
                    melSpec[mel, frame] = Mathf.Clamp(melSpec[mel, frame], -1f, 1f);
                }
            }
            
            return melSpec;
        }

        /// <summary>
        /// Pads mel spectrogram to target frame count for consistent tensor dimensions.
        /// This method extends the mel spectrogram to a standardized length by padding
        /// with zeros, ensuring consistent input dimensions for neural network processing.
        /// </summary>
        /// <param name="melSpec">The input mel spectrogram to pad</param>
        /// <param name="currentFrames">The current number of frames in the spectrogram</param>
        /// <returns>A padded mel spectrogram with TARGET_FRAMES columns</returns>
        private static float[,] PadMelSpectrogram(float[,] melSpec, int currentFrames)
        {
            float[,] paddedMel = new float[N_MELS, TARGET_FRAMES];
            for (int mel = 0; mel < N_MELS; mel++)
            {
                for (int frame = 0; frame < TARGET_FRAMES; frame++)
                {
                    paddedMel[mel, frame] = frame < currentFrames ? melSpec[mel, frame] : 0f;
                }
            }
            return paddedMel;
        }

        #endregion

        #region Private Methods - Mel Filterbank Creation

        /// <summary>
        /// Creates a librosa-compatible mel filterbank for frequency-to-mel conversion.
        /// This method generates triangular mel filters with proper normalization matching
        /// librosa's mel filterbank implementation exactly for consistent audio processing.
        /// </summary>
        /// <param name="nFft">The FFT size used for frequency analysis</param>
        /// <returns>A 2D array containing mel filter coefficients [mel_bands, frequency_bins]</returns>
        private static float[,] CreateLibrosaMelFilterBank(int nFft)
        {
            int nFreqBins = nFft / 2 + 1;
            float[,] melFilters = new float[N_MELS, nFreqBins];
            
            // Create mel frequency points (matching librosa exactly)
            float melMin = HzToMel(0f);
            float melMax = HzToMel(8000f); // Nyquist frequency for 16kHz
            
            float[] melPoints = new float[N_MELS + 2];
            for (int i = 0; i < melPoints.Length; i++)
            {
                float mel = melMin + (melMax - melMin) * i / (melPoints.Length - 1);
                melPoints[i] = MelToHz(mel);
            }
            
            // Convert mel frequencies to FFT bin indices
            float[] binPoints = new float[N_MELS + 2];
            for (int i = 0; i < binPoints.Length; i++)
            {
                binPoints[i] = melPoints[i] * nFft / SAMPLE_RATE;
            }
            
            // Create triangular filters with proper normalization matching librosa
            for (int mel = 0; mel < N_MELS; mel++)
            {
                float leftBin = binPoints[mel];
                float centerBin = binPoints[mel + 1];
                float rightBin = binPoints[mel + 2];
                
                // Calculate normalization factors for triangular filter slopes
                float leftWidth = centerBin - leftBin;
                float rightWidth = rightBin - centerBin;
                
                for (int bin = 0; bin < nFreqBins; bin++)
                {
                    if (bin >= leftBin && bin <= centerBin && leftWidth > 0)
                    {
                        // Ascending slope with librosa normalization: 2.0 / (rightBin - leftBin)
                        melFilters[mel, bin] = 2.0f * (bin - leftBin) / ((rightBin - leftBin) * leftWidth);
                    }
                    else if (bin > centerBin && bin <= rightBin && rightWidth > 0)
                    {
                        // Descending slope with librosa normalization: 2.0 / (rightBin - leftBin)
                        melFilters[mel, bin] = 2.0f * (rightBin - bin) / ((rightBin - leftBin) * rightWidth);
                    }
                    else
                    {
                        melFilters[mel, bin] = 0f;
                    }
                }
            }
            
            return melFilters;
        }

        #endregion

        #region Private Methods - Frequency Scale Conversion

        /// <summary>
        /// Converts frequency from Hz to mel scale using the standard mel scale formula.
        /// This matches the librosa implementation for consistent frequency scaling across platforms.
        /// The mel scale is a perceptual scale of pitches judged by listeners to be equal in distance.
        /// </summary>
        /// <param name="hz">The frequency in Hz to convert</param>
        /// <returns>The corresponding frequency in mel scale</returns>
        private static float HzToMel(float hz)
        {
            return 2595f * Mathf.Log10(1f + hz / 700f);
        }
        
        /// <summary>
        /// Converts frequency from mel scale to Hz using the inverse mel scale formula.
        /// This matches the librosa implementation for consistent frequency scaling across platforms.
        /// Used to convert mel-spaced filter centers back to linear frequency for filter creation.
        /// </summary>
        /// <param name="mel">The frequency in mel scale to convert</param>
        /// <returns>The corresponding frequency in Hz</returns>
        private static float MelToHz(float mel)
        {
            return 700f * (Mathf.Pow(10f, mel / 2595f) - 1f);
        }

        #endregion
    }
}
