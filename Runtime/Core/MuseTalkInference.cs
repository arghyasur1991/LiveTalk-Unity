using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace LiveTalk.Core
{
    using System.IO;
    using API;
    using Utils;

    /// <summary>
    /// Core MuseTalk inference engine that manages ONNX models for real-time talking head generation.
    /// Provides comprehensive functionality for avatar processing, audio feature extraction, and synchronized
    /// video frame generation using advanced neural network models.
    /// </summary>
    internal class MuseTalkInference : IDisposable
    {
        #region Private Fields
        private static readonly string MODEL_RELATIVE_PATH = "MuseTalk";
        
        // ONNX Models
        private Model _unet;
        private Model _vaeEncoder;
        private Model _vaeDecoder;
        private Model _positionalEncoding;
        private WhisperModel _whisperModel;
        private FaceAnalysis _faceAnalysis;
        
        // Configuration and State
        private readonly LiveTalkConfig _config;
        private bool _generatorModelsInitialized = false;
        private bool _preprocessorModelsInitialized = false;
        private bool _disposed = false;
        
        // Performance Optimization
        private float[] _reusableBatchArray = new float[0];

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the MuseTalkInference class with the specified configuration.
        /// </summary>
        /// <param name="config">The LiveTalk configuration containing model paths and inference settings</param>
        /// <exception cref="ArgumentNullException">Thrown when config is null</exception>
        public MuseTalkInference(LiveTalkConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            InitializeAllModels();
        }

        #endregion

        #region Public Methods
        
        public async Task<AvatarData> ProcessAvatarImages(List<string> avatarFramePaths)
        {
            if (avatarFramePaths == null)
                throw new ArgumentNullException(nameof(avatarFramePaths));
            if (avatarFramePaths.Count == 0)
                throw new ArgumentException("Avatar frame paths list cannot be empty", nameof(avatarFramePaths));

            Logger.LogVerbose($"[MuseTalkInference] Processing {avatarFramePaths.Count} avatar frame paths");

            AvatarData avatarData = new();
            await _faceAnalysis.StartFaceAnalysisSession();
            await _faceAnalysis.StartFaceParsingSession();
            await _vaeEncoder.StartSession();
            foreach (var path in avatarFramePaths)
            {
                var texture = FileUtils.LoadFrame(path);
                await ComputeAvatarDataForFrame(texture, avatarData);
            }
            _faceAnalysis.EndFaceAnalysisSession();
            _faceAnalysis.EndFaceParsingSession();
            _vaeEncoder.EndSession();

            Logger.LogVerbose($"[MuseTalkInference] Successfully processed {avatarData.FaceRegions.Count} " + 
                $"face regions with {avatarData.Latents.Count} latent sets across {avatarFramePaths.Count} avatar textures");

            return avatarData;
        }
        
        /// <summary>
        /// Asynchronously processes avatar images to extract face regions, landmarks, and latent representations.
        /// This method performs face detection, segmentation mask generation, and VAE encoding to prepare
        /// the avatar data for subsequent video generation.
        /// </summary>
        /// <param name="avatarTextures">The list of avatar texture images to process</param>
        /// <returns>A task containing the processed avatar data with face regions and latent vectors</returns>
        /// <exception cref="ArgumentNullException">Thrown when avatarTextures is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when no faces are detected or latent generation fails</exception>
        public async Task<AvatarData> ProcessAvatarImages(List<Texture2D> avatarTextures)
        {
            if (avatarTextures == null)
                throw new ArgumentNullException(nameof(avatarTextures));
            if (avatarTextures.Count == 0)
                throw new ArgumentException("Avatar textures list cannot be empty", nameof(avatarTextures));

            Logger.LogVerbose($"[MuseTalkInference] Processing {avatarTextures.Count} avatar textures");
            
            var avatarData = new AvatarData();
            await _faceAnalysis.StartFaceAnalysisSession();
            await _faceAnalysis.StartFaceParsingSession();
            await _vaeEncoder.StartSession();

            foreach (var texture in avatarTextures)
            {
                await ComputeAvatarDataForFrame(texture, avatarData);
            }

            _faceAnalysis.EndFaceAnalysisSession();
            _faceAnalysis.EndFaceParsingSession();
            _vaeEncoder.EndSession();

            Logger.LogVerbose($"[MuseTalkInference] Successfully processed {avatarData.FaceRegions.Count} " + 
                $"face regions with {avatarData.Latents.Count} latent sets across {avatarTextures.Count} avatar textures");            
            return avatarData;
        }

        /// <summary>
        /// Asynchronously generates talking head video frames from avatar textures and audio input.
        /// This method processes avatar images, extracts audio features, and streams generated frames
        /// to the output stream using Unity coroutines for non-blocking execution.
        /// </summary>
        /// <param name="avatarTextures">The list of avatar texture images to animate</param>
        /// <param name="audioClip">The audio clip to synchronize with the generated video</param>
        /// <param name="stream">The output stream to receive generated video frames</param>
        /// <returns>An enumerator for Unity coroutine execution</returns>
        /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
        public IEnumerator GenerateAsync(List<Texture2D> avatarTextures, AudioClip audioClip, FrameStream stream)
        {
            if (avatarTextures == null)
                throw new ArgumentNullException(nameof(avatarTextures));
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            var avatarTask = ProcessAvatarImages(avatarTextures);
            yield return new WaitUntil(() => avatarTask.IsCompleted);
            var avatarData = avatarTask.Result;

            yield return GenerateWithPreloadedDataAsync(audioClip, avatarData, stream);
        }

        /// <summary>
        /// Asynchronously generates talking head video frames using preloaded avatar data and audio input.
        /// This optimized method bypasses avatar processing and directly generates frames from
        /// previously processed avatar data, improving performance for repeated generations.
        /// </summary>
        /// <param name="audioClip">The audio clip to synchronize with the generated video</param>
        /// <param name="avatarData">The preloaded avatar data containing face regions and latents</param>
        /// <param name="stream">The output stream to receive generated video frames</param>
        /// <returns>An enumerator for Unity coroutine execution</returns>
        /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
        public IEnumerator GenerateWithPreloadedDataAsync(
            AudioClip audioClip, 
            AvatarData avatarData, 
            FrameStream stream)
        {
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
            if (avatarData == null)
                throw new ArgumentNullException(nameof(avatarData));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            
            // Step 1: Process audio and extract features
            var audioTask = ProcessAudio(audioClip);
            yield return new WaitUntil(() => audioTask.IsCompleted);
            var audioFeatures = audioTask.Result;
            stream.TotalExpectedFrames = audioFeatures.FeatureChunks.Count;
            
            // Step 2: Generate and stream video frames
            yield return GenerateFramesStreaming(avatarData, audioFeatures, stream);
            
            stream.Finished = true;
        }

        #endregion

        #region Private Methods - Model Initialization
        
        /// <summary>
        /// Initializes all ONNX models required for both preprocessing and generation phases.
        /// This method ensures all models are loaded before starting the inference pipeline.
        /// </summary>
        private void InitializeAllModels()
        {
            InitializeGeneratorModels();
            InitializePreprocessorModels();
        }

        /// <summary>
        /// Starts the session for the generator models
        /// </summary>
        private async Task StartGeneratorSession()
        {
            await _unet.StartSession();
            await _vaeDecoder.StartSession();
            await _positionalEncoding.StartSession();
            // Whisper model is started in ProcessAudio()
        }

        /// <summary>
        /// Ends the session for the generator models
        /// </summary>
        private void EndGeneratorSession()
        {
            _unet.EndSession();
            _vaeDecoder.EndSession();
            _positionalEncoding.EndSession();
            // Whisper model is ended in ProcessAudio()
            _reusableBatchArray = null;
        }

        /// <summary>
        /// Initializes the models required for video frame generation.
        /// Includes UNet, VAE decoder, Whisper model, and positional encoding components.
        /// </summary>
        private void InitializeGeneratorModels()
        {
            if (_generatorModelsInitialized)
                return;
            
            bool forceFP32 = _config.MemoryUsage == MemoryUsage.Quality;
            Precision fp16Precision = forceFP32 ? Precision.FP32 : Precision.FP16;

            Logger.LogVerbose("[MuseTalkInference] Initializing generator models...");
            _unet = new Model(_config, "unet", MODEL_RELATIVE_PATH, ExecutionProvider.CoreML, fp16Precision);
            if (_config.MemoryUsage == MemoryUsage.Optimal)
            {
                // Unet needs to be kept alive for performance reasons as loading is very slow
                _unet.SetLoadPolicy(Model.ModelLoadPolicy.OnDemandKeepAlive);
            }
            _vaeDecoder = new Model(_config, "vae_decoder", MODEL_RELATIVE_PATH, ExecutionProvider.CoreML, fp16Precision);
            _whisperModel = new WhisperModel(_config);
            _positionalEncoding = new Model(_config, "positional_encoding", MODEL_RELATIVE_PATH);
            _generatorModelsInitialized = true;
        }

        /// <summary>
        /// Initializes the models required for avatar preprocessing.
        /// Includes VAE encoder for latent generation and face analysis for detection/segmentation.
        /// </summary>
        private void InitializePreprocessorModels()
        {
            if (_preprocessorModelsInitialized)
                return;
                
            Logger.LogVerbose("[MuseTalkInference] Initializing preprocessor models...");
            _vaeEncoder = new Model(_config, "vae_encoder", MODEL_RELATIVE_PATH, ExecutionProvider.CoreML, Precision.FP16);
            _faceAnalysis = FaceAnalysis.CreateOrGetInstance(_config);
            _preprocessorModelsInitialized = true;
        }

        #endregion

        #region Private Methods - Face Segmentation

        /// <summary>
        /// Crops a face region from the input frame with version-specific margin adjustments.
        /// This method extracts the face area defined by the bounding box, applies version-specific margins
        /// (such as extra bottom margin for v15), and resizes to standard dimensions for processing.
        /// </summary>
        /// <param name="frame">The input frame containing the face to crop</param>
        /// <param name="bbox">The face bounding box coordinates (x1, y1, x2, y2)</param>
        /// <param name="version">The version identifier that determines specific margin adjustments (e.g., "v15")</param>
        /// <returns>A cropped and resized face frame (256x256) ready for downstream processing</returns>
        /// <exception cref="Exception">Thrown when crop dimensions are invalid</exception>
        private Frame CropFaceRegion(Frame frame, Vector4 bbox, string version)
        {
            if (frame.data == null)
                return new Frame(null, 0, 0);
                
            int x1 = Mathf.RoundToInt(bbox.x);
            int y1 = Mathf.RoundToInt(bbox.y);
            int x2 = Mathf.RoundToInt(bbox.z);
            int y2 = Mathf.RoundToInt(bbox.w);
            
            // Add version-specific margin
            if (version == "v15")
            {
                y2 += 10; // extra margin for v15
                y2 = Mathf.Min(y2, frame.height);
            }
            
            int cropWidth = x2 - x1;
            int cropHeight = y2 - y1;
            
            if (frame.width <= 0 || frame.height <= 0)
            {
                Logger.LogError($"[FaceAnalysis] Invalid crop dimensions: {cropWidth}x{cropHeight}");
                throw new Exception($"[FaceAnalysis] Invalid crop dimensions: {cropWidth}x{cropHeight}");
            }

            // Extract face region (matching InsightFaceHelper coordinate system)
            var croppedFrame = FrameUtils.CropFrame(frame, new Rect(x1, y1, cropWidth, cropHeight));
            
            // Resize to standard size (256x256 for MuseTalk, matching InsightFaceHelper)
            var resizedFrame = FrameUtils.ResizeFrame(croppedFrame, 256, 256, SamplingMode.Bilinear);            
            return resizedFrame;
        }
        
        /// <summary>
        /// Asynchronously pre-computes segmentation data that can be cached and reused for all frames.
        /// This includes face_large crop, BiSeNet segmentation mask, and all blending masks
        /// to optimize performance during frame generation.
        /// </summary>
        /// <param name="originalImage">The original input image containing the face</param>
        /// <param name="faceBbox">The bounding box coordinates of the detected face</param>
        /// <returns>A task containing the precomputed segmentation data</returns>
        /// <exception cref="InvalidOperationException">Thrown when segmentation mask generation fails</exception>
        private async Task<SegmentationData> ComputeSegmentationData(Frame originalImage, Vector4 faceBbox)
        {
            // Apply version-specific adjustments to face bbox (matching BlendFaceWithOriginal logic)
            Vector4 adjustedFaceBbox = faceBbox;
            if (_config.Version == "v15") // v15 mode
            {
                // Apply v15 extra margin to y2 (bottom of face bbox)
                adjustedFaceBbox.w = Mathf.Min(adjustedFaceBbox.w + _config.ExtraMargin, originalImage.height);
            }
            
            // Calculate expanded crop box for face_large
            var cropRect = MathUtils.ExpandBoundingBox(adjustedFaceBbox, 1.5f); // expandFactor = 1.5f
            var cropBox = new Vector4(cropRect.x, cropRect.y, cropRect.x + cropRect.width, cropRect.y + cropRect.height);
            
            // Create face_large crop
            var faceLarge = FrameUtils.CropFrame(originalImage, cropRect);
            
            // Generate face segmentation mask using BiSeNet on face_large
            var segmentationMask = await GenerateFaceSegmentationMask(faceLarge);
            
            if (segmentationMask.data == null)
                throw new InvalidOperationException("Failed to generate segmentation mask");
            
            // OPTIMIZATION: Precompute all blending masks that are independent of faceTexture
            // These masks only depend on segmentation, face bbox, and crop box - all available now
            
            // Step 1: Create mask_small by cropping BiSeNet mask to face bbox
            var maskSmall = ImageBlendingHelper.CreateSmallMask(segmentationMask, adjustedFaceBbox, cropRect);
            
            // Step 2: Create full mask by pasting mask_small back into face_large dimensions
            var fullMask = ImageBlendingHelper.CreateFullMask(segmentationMask, maskSmall, adjustedFaceBbox, cropRect);
            
            // Step 3: Apply upper boundary ratio to preserve upper face
            const float upperBoundaryRatio = 0.5f; // Standard value used in ApplySegmentationMask
            var boundaryMask = ImageBlendingHelper.ApplyUpperBoundaryRatio(fullMask, upperBoundaryRatio);
            
            // Step 4: Apply Gaussian blur for smooth blending
            var blurredMask = ImageBlendingHelper.ApplyGaussianBlurToMask(boundaryMask);
            
            return new SegmentationData
            {
                FaceLarge = faceLarge,
                SegmentationMask = segmentationMask,
                AdjustedFaceBbox = adjustedFaceBbox,
                CropBox = cropBox,
                MaskSmall = maskSmall,
                FullMask = fullMask,                
                BoundaryMask = boundaryMask,
                BlurredMask = blurredMask
            };
        }
        
        /// <summary>
        /// Asynchronously generates and caches a face segmentation mask for the specified face region.
        /// Uses morphological operations to create a precise mask for face blending.
        /// </summary>
        /// <param name="faceLarge">The face region image to generate a mask for</param>
        /// <returns>A task containing the generated segmentation mask</returns>
        /// <exception cref="InvalidOperationException">Thrown when face segmentation fails</exception>
        private async Task<Frame> GenerateFaceSegmentationMask(Frame faceLarge)
        {
            try
            {
                var mask = await _faceAnalysis.GenerateFaceSegmentationMask(faceLarge, "jaw");
                if (mask.data != null)
                {
                    // Resize to target dimensions if needed
                    if (mask.width != faceLarge.width || mask.height != faceLarge.height)
                    {
                        var resizedMask = FrameUtils.ResizeFrame(mask, faceLarge.width, faceLarge.height, SamplingMode.Bilinear);
                        return resizedMask;
                    }
                    return mask;
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkInference] ONNX face parsing failed: {e.Message}");
            }
            
            throw new InvalidOperationException("Face segmentation failed and no fallback is available");
        }
        
        #endregion

        #region Private Methods - Audio Processing
        
        /// <summary>
        /// Asynchronously processes an audio clip and extracts Whisper features for synchronization.
        /// Handles stereo to mono conversion and validates the Whisper model initialization.
        /// </summary>
        /// <param name="audioClip">The audio clip to process</param>
        /// <returns>A task containing the extracted audio features</returns>
        /// <exception cref="ArgumentNullException">Thrown when audioClip is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when Whisper model is not initialized</exception>
        private async Task<AudioFeatures> ProcessAudio(AudioClip audioClip)
        {
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
                
            // Convert AudioClip to float array
            var audioData = AudioUtils.AudioClipToFloatArray(audioClip);
            
            // Convert stereo to mono if needed
            if (audioClip.channels == 2)
            {
                audioData = AudioUtils.StereoToMono(audioData);
            }

            await _whisperModel.StartSession();
            
            if (_whisperModel == null || !_whisperModel.IsInitialized)
            {
                throw new InvalidOperationException("Whisper model is not initialized. Real Whisper model is required for proper inference.");
            }
            
            // Use ONNX Whisper model
            var features = await ExtractWhisperFeatures(audioData, audioClip.frequency);
            _whisperModel.EndSession();
            return features;
        }
        
        /// <summary>
        /// Asynchronously extracts Whisper features from raw audio data using the ONNX WhisperModel.
        /// </summary>
        /// <param name="audioData">The raw audio data as float array</param>
        /// <param name="sampleRate">The sample rate of the audio data</param>
        /// <returns>A task containing the extracted audio features</returns>
        /// <exception cref="InvalidOperationException">Thrown when Whisper model fails to process audio</exception>
        private async Task<AudioFeatures> ExtractWhisperFeatures(float[] audioData, int sampleRate)
        {
            return await Task.Run(async () =>
            {
                var features = await _whisperModel.ProcessAudio(audioData, sampleRate) ?? 
                    throw new InvalidOperationException("Whisper model failed to process audio. Check model loading and input data.");
                return features;
            });
        }

        #endregion

        #region Private Methods - Generator Processing

        /// <summary>
        /// Asynchronously generates video frames using UNet and VAE decoder with streaming output.
        /// This method yields frames as they're generated for real-time feedback, similar to LivePortrait.
        /// Uses cyclic latent animation and frame-by-frame processing for optimal performance.
        /// </summary>
        /// <param name="avatarData">The processed avatar data containing face regions and latents</param>
        /// <param name="audioFeatures">The extracted audio features for synchronization</param>
        /// <param name="stream">The output stream to receive generated frames</param>
        /// <returns>An enumerator for Unity coroutine execution</returns>
        /// <exception cref="InvalidOperationException">Thrown when no avatar latents are available</exception>
        private IEnumerator GenerateFramesStreaming(AvatarData avatarData, AudioFeatures audioFeatures, FrameStream stream)
        {
            // Use audio length to determine frame count
            int numFrames = audioFeatures.FeatureChunks.Count;
            
            if (avatarData.Latents.Count == 0)
            {
                Logger.LogError("[MuseTalkInference] No avatar latents available for frame generation");
                yield break;
            }
            Logger.LogVerbose($"[MuseTalkInference] Processing {numFrames} frames");

            var startSessionTask = StartGeneratorSession();
            yield return new WaitUntil(() => startSessionTask.IsCompleted);

            // Create cycled latent list for smooth animation
            var cycleDLatents = new List<float[]>(avatarData.Latents);
            var reversedLatents = new List<float[]>(avatarData.Latents);
            reversedLatents.Reverse();
            cycleDLatents.AddRange(reversedLatents);
            
            for (int idx = 0; idx < numFrames; idx++)
            {
                // Prepare batch data using the frame index to cycle through latents
                var latentBatch = PrepareLatentBatchWithCycling(cycleDLatents, idx);
                var audioBatch = PrepareAudioBatch(audioFeatures.FeatureChunks, idx);
                
                // Add positional encoding to audio (async)
                var audioWithPETask = AddPositionalEncoding(audioBatch);
                yield return new WaitUntil(() => audioWithPETask.IsCompleted);
                var audioWithPE = audioWithPETask.Result;
                
                // Run UNet inference (async)
                var predictedLatentsTask = RunUNet(latentBatch, audioWithPE);
                yield return new WaitUntil(() => predictedLatentsTask.IsCompleted);
                var predictedLatents = predictedLatentsTask.Result;
                
                // Decode latents to images (async)
                var batchFramesTask = DecodeLatents(predictedLatents, idx, avatarData);
                yield return new WaitUntil(() => batchFramesTask.IsCompleted);
                var frame = batchFramesTask.Result;
                
                // Stream frames to output as they're generated
                stream.Queue.Enqueue(TextureUtils.FrameToTexture2D(frame));
                
                // Log batch performance
                Logger.LogVerbose($"[MuseTalkInference] Frame {idx} completed");
            }

            EndGeneratorSession();
        }

        /// <summary>
        /// Asynchronously computes avatar data for a single frame.
        /// This method detects faces, crops the face region, generates segmentation masks,
        /// and computes latents for the UNet model.
        /// </summary>
        /// <param name="texture"></param>
        /// <param name="avatarData"></param>
        /// <returns></returns>
        private async Task ComputeAvatarDataForFrame(Texture2D texture, AvatarData avatarData)
        {
            var frame = TextureUtils.Texture2DToFrame(texture);
            UnityEngine.Object.DestroyImmediate(texture);
            await Task.Run(async () =>
            {
                var bbox = await _faceAnalysis.GetLandmarkAndBbox(frame);
                if (bbox == Vector4.zero)
                {
                    Logger.LogWarning($"[MuseTalkInference] No face detected in image {texture.width}x{texture.height}");
                    return;
                }

                var croppedFrame = CropFaceRegion(frame, bbox, _config.Version);
                var segmentationData = await ComputeSegmentationData(frame, bbox);
                var faceData = new FaceData
                {
                    HasFace = true,
                    BoundingBox = new Rect(bbox.x, bbox.y, bbox.z - bbox.x, bbox.w - bbox.y),
                    CroppedFaceTexture = croppedFrame,
                    OriginalTexture = frame,
                    FaceLarge = segmentationData.FaceLarge,
                    SegmentationMask = segmentationData.SegmentationMask,
                    AdjustedFaceBbox = segmentationData.AdjustedFaceBbox,
                    CropBox = segmentationData.CropBox,
                    MaskSmall = segmentationData.MaskSmall,
                    FullMask = segmentationData.FullMask,
                    BoundaryMask = segmentationData.BoundaryMask,
                    BlurredMask = segmentationData.BlurredMask
                };
                avatarData.FaceRegions.Add(faceData);

                var latents = await GetLatentsForUNet(croppedFrame);
                avatarData.Latents.Add(latents);
            });
        }

        /// <summary>
        /// Asynchronously encodes an image using the VAE encoder to generate latent representations.
        /// If applyLowerHalfMask is true, the lower half of the image is masked.
        /// Resizes the image to 256x256 and applies normalization before encoding.
        /// </summary>
        /// <param name="image">The image frame to encode</param>
        /// <param name="applyLowerHalfMask">Whether to apply a lower half mask to the image</param>
        /// <returns>A task containing the encoded latent array</returns>
        /// <exception cref="InvalidOperationException">Thrown when VAE encoding fails</exception>
        private async Task<float[]> EncodeImage(Frame image, bool applyLowerHalfMask = false)
        {
            // Resize image data to 256x256 for VAE encoder
            var resizedData = FrameUtils.ResizeFrame(image, 256, 256, SamplingMode.Bilinear);
            var inputTensor = FrameUtils.FrameToTensor(resizedData, 2.0f / 255.0f, -1.0f, applyLowerHalfMask);
            
            var inputs = new List<Tensor<float>>
            {
                inputTensor
            };
            
            // Run VAE encoder
            var results = await _vaeEncoder.Run(inputs);
            var latents = results.First(r => r.Name == "latents").AsTensor<float>();
            var result = latents.ToArray();
            
            return result;
        }

        /// <summary>
        /// Asynchronously generates latents for UNet inference by combining masked and reference latents.
        /// </summary>
        /// <param name="image">The image frame to generate latents for</param>
        /// <returns>A task containing the combined latent array for UNet processing</returns>
        /// <exception cref="InvalidOperationException">Thrown when latent generation fails</exception>
        private async Task<float[]> GetLatentsForUNet(Frame image)
        {
            // Get masked latents (lower half masked)
            var maskedLatents = await EncodeImage(image, true);
            
            // Get reference latents (full image)
            var refLatents = await EncodeImage(image);
            
            if (maskedLatents.Length != refLatents.Length)
            {
                throw new InvalidOperationException("Masked and reference latents must have same size");
            }
            
            const int batch = 1;
            const int maskedChannels = 4;
            const int refChannels = 4;
            const int totalChannels = maskedChannels + refChannels;
            const int latentHeight = 32;
            const int latentWidth = 32;
            const int spatialSize = latentHeight * latentWidth;
            
            var combinedLatents = new float[batch * totalChannels * latentHeight * latentWidth];
            
            unsafe
            {
                fixed (float* maskedPtr = maskedLatents)
                fixed (float* refPtr = refLatents)
                fixed (float* combinedPtr = combinedLatents)
                {
                    // Copy channel by channel using fast memory operations
                    for (int b = 0; b < batch; b++)
                    {
                        // Copy masked latents channels (0-3) using Buffer.MemoryCopy
                        for (int c = 0; c < maskedChannels; c++)
                        {
                            int srcOffset = b * maskedChannels * spatialSize + c * spatialSize;
                            int dstOffset = b * totalChannels * spatialSize + c * spatialSize;
                            
                            Buffer.MemoryCopy(
                                maskedPtr + srcOffset,
                                combinedPtr + dstOffset,
                                spatialSize * sizeof(float),
                                spatialSize * sizeof(float)
                            );
                        }
                        
                        // Copy reference latents channels (4-7) using Buffer.MemoryCopy
                        for (int c = 0; c < refChannels; c++)
                        {
                            int srcOffset = b * refChannels * spatialSize + c * spatialSize;
                            int dstOffset = b * totalChannels * spatialSize + (maskedChannels + c) * spatialSize;
                            
                            Buffer.MemoryCopy(
                                refPtr + srcOffset,
                                combinedPtr + dstOffset,
                                spatialSize * sizeof(float),
                                spatialSize * sizeof(float)
                            );
                        }
                    }
                }
            }
            
            return combinedLatents;
        }
        
        /// <summary>
        /// Prepares a latent batch with proper cycling for smooth frame-based animation.
        /// </summary>
        /// <param name="cycleDLatents">The cycled latent arrays for animation smoothness</param>
        /// <param name="startIdx">The starting frame index for latent selection</param>
        /// <returns>A properly formatted tensor batch for UNet processing with dimensions [1, 8, 32, 32]</returns>
        /// <exception cref="InvalidOperationException">Thrown when latent size doesn't match expected dimensions</exception>
        private DenseTensor<float> PrepareLatentBatchWithCycling(List<float[]> cycleDLatents, int startIdx)
        {
            const int channels = 8, height = 32, width = 32;
            int totalSize = channels * height * width;
            var flatBatch = new float[totalSize];
            int globalFrameIdx = startIdx;
            var latentIdx = globalFrameIdx % cycleDLatents.Count;
            var latent = cycleDLatents[latentIdx];
            
            // Verify latent size
            const int expectedLatentSize = 1 * channels * height * width;
            if (latent.Length != expectedLatentSize)
            {
                throw new InvalidOperationException($"Latent size mismatch: got {latent.Length}, expected {expectedLatentSize}");
            }
            
            // OPTIMIZATION 4: Use unsafe copy for latent batch preparation
            const int singleLatentSize = channels * height * width;
            
            unsafe
            {
                fixed (float* src = latent)
                fixed (float* dst = flatBatch)
                {
                    Buffer.MemoryCopy(src, dst, singleLatentSize * sizeof(float), singleLatentSize * sizeof(float));
                }
            }
            
            // Return tensor with proper batch dimension: [batchSize, 8, 32, 32]
            return new DenseTensor<float>(flatBatch, new[] { 1, channels, height, width });
        }
        
        /// <summary>
        /// Prepares an audio feature batch for UNet processing from the audio chunk list.
        /// Formats the audio features into the expected tensor dimensions for neural network input.
        /// </summary>
        /// <param name="audioChunks">The list of audio feature chunks extracted from Whisper</param>
        /// <param name="startIdx">The starting index to select the audio chunk for this frame</param>
        /// <returns>A formatted audio tensor with dimensions [1, 50, 384] for UNet processing</returns>
        private DenseTensor<float> PrepareAudioBatch(List<float[]> audioChunks, int startIdx)
        {
            int timeSteps = 50, features = 384;
            int totalSize = timeSteps * features;
            var flatBatch = new float[totalSize];
            var audioIdx = startIdx;
            if (audioIdx < audioChunks.Count)
            {
                var chunk = audioChunks[audioIdx];
                int batchOffset = 0;
                for (int idx = 0; idx < chunk.Length && idx < timeSteps * features; idx++)
                {
                    flatBatch[batchOffset + idx] = chunk[idx];
                }
            }
            
            return new DenseTensor<float>(flatBatch, new[] { 1, timeSteps, features });
        }
        
        /// <summary>
        /// Asynchronously adds positional encoding to audio features for temporal awareness.
        /// This enhances the audio features with positional information that helps the UNet
        /// understand the temporal relationships in the audio sequence.
        /// </summary>
        /// <param name="audioBatch">The audio feature batch tensor to encode</param>
        /// <returns>A task containing the audio features enhanced with positional encoding</returns>
        /// <exception cref="InvalidOperationException">Thrown when positional encoding model fails</exception>
        private async Task<DenseTensor<float>> AddPositionalEncoding(DenseTensor<float> audioBatch)
        {
            return await Task.Run(async () =>
            {
                var inputs = new List<Tensor<float>>
                {
                    audioBatch
                };
                
                using var results = await _positionalEncoding.RunDisposable(inputs);
                var output = results.First().AsTensor<float>();
                var result = new DenseTensor<float>(output.ToArray(), output.Dimensions.ToArray());
                
                return result;
            });
        }
        
        /// <summary>
        /// Asynchronously runs UNet inference to predict new latents from audio and image inputs.
        /// </summary>
        /// <param name="latentBatch">The prepared latent batch tensor from avatar images</param>
        /// <param name="audioBatch">The prepared audio batch tensor with positional encoding</param>
        /// <returns>A task containing the predicted latent tensor for VAE decoding</returns>
        /// <exception cref="InvalidOperationException">Thrown when UNet inference fails</exception>
        private async Task<Tensor<float>> RunUNet(DenseTensor<float> latentBatch, DenseTensor<float> audioBatch)
        {
            return await Task.Run(async () =>
            {
                var inputs = new List<Tensor<float>>
                {
                    latentBatch,
                    audioBatch
                };
                
                // Keep the result alive as a member variable to prevent GC of tensor memory
                var outputs = await _unet.Run(inputs);
                var output = outputs.First().AsTensor<float>();
                
                // OPTIMIZATION: Return the tensor directly without copying via ToArray()
                // The tensor memory is kept alive by _reusableUNetResult member variable
                Tensor<float> result;
                if (output is DenseTensor<float> denseTensor)
                {
                    // Create a new DenseTensor that shares the same Memory<float> buffer (zero copy!)
                    // Memory remains valid because _reusableUNetResult keeps it alive
                    result = new DenseTensor<float>(denseTensor.Buffer, denseTensor.Dimensions.ToArray());
                }
                else
                {
                    // Fallback: only copy if we can't access the buffer directly
                    Logger.LogVerbose($"[MuseTalkInference] UNet_ONNX_Inference: Fallback to ToArray()");
                    result = new DenseTensor<float>(output.ToArray(), output.Dimensions.ToArray());
                }
                
                return result;
            });
        }
        
        /// <summary>
        /// Asynchronously decodes latents back to images using VAE decoder and applies seamless blending.
        /// This method combines the generated face texture with the original avatar image using precomputed
        /// segmentation masks for natural-looking results. Optimized with reusable arrays and zero-copy tensor operations.
        /// </summary>
        /// <param name="unetOutputBatch">The predicted latent tensor from UNet inference</param>
        /// <param name="globalStartIdx">The global frame index for proper avatar cycling</param>
        /// <param name="avatarData">The avatar data containing original images and precomputed blending masks</param>
        /// <returns>A task containing the final blended texture ready for display</returns>
        /// <exception cref="InvalidOperationException">Thrown when VAE decoding or blending fails</exception>
        private async Task<Frame> DecodeLatents(Tensor<float> unetOutputBatch, int globalStartIdx, AvatarData avatarData)
        {      
            return await Task.Run(async () =>
            {
                var tensors = new List<Tensor<float>>();
                Tensor<float> batchImageOutput = null; // Keep reference alive
                
                try
                {
                    var inputs = new List<Tensor<float>>
                    {
                        unetOutputBatch
                    };
                    
                    var outputs = await _vaeDecoder.Run(inputs);
                    var batchImageOutputValue = outputs.First();
                    batchImageOutput = batchImageOutputValue.AsTensor<float>();
                    
                    var imageDims = new int[batchImageOutput.Dimensions.Length - 1]; // Remove batch dimension
                    for (int i = 1; i < batchImageOutput.Dimensions.Length; i++)
                    {
                        imageDims[i - 1] = (int)batchImageOutput.Dimensions[i];
                    }
                    
                    int imagesPerBatch = imageDims.Aggregate(1, (a, b) => a * b);
                    int totalElements = imagesPerBatch;
                    
                    if (_reusableBatchArray.Length < totalElements)
                    {
                        _reusableBatchArray = new float[totalElements];
                    }
                    
                    if (batchImageOutputValue.Value is DenseTensor<float> denseTensor)
                    {
                        // Use DenseTensor.Buffer for direct access
                        denseTensor.Buffer.Span.CopyTo(_reusableBatchArray.AsSpan(0, totalElements));
                    }
                    else
                    {
                        // Fallback to ToArray() if not DenseTensor
                        var sourceArray = batchImageOutput.ToArray();
                        sourceArray.CopyTo(_reusableBatchArray.AsSpan(0, totalElements));
                    }
                    
                    // Create Memory<float> that references the reusable array directly (no copying!)
                    var imageMemory = _reusableBatchArray.AsMemory(0, imagesPerBatch);
                    var singleImageTensor = new DenseTensor<float>(imageMemory, imageDims);
                    tensors.Add(singleImageTensor);
                }
                catch (Exception e)
                {
                    Logger.LogWarning($"[MuseTalkInference] Batch VAE decoding failed: {e.Message}");
                    throw;
                }
                var tensor = tensors[0];
                
                // Calculate global frame index for proper numbering
                int globalFrameIdx = globalStartIdx;
                
                // Step 1: Convert tensor to raw decoded texture
                var rawDecodedTexture = FrameUtils.TensorToFrame(tensor);
                
                // Step 2: Resize to face crop dimensions
                if (avatarData != null && avatarData.FaceRegions.Count > 0)
                {
                    // Get corresponding face bbox for sizing
                    int avatarIndex = globalFrameIdx % (2 * avatarData.FaceRegions.Count); // cycled latents
                    if (avatarIndex >= avatarData.FaceRegions.Count)
                    {
                        avatarIndex = (2 * avatarData.FaceRegions.Count) - 1 - avatarIndex;
                    }
                    var faceData = avatarData.FaceRegions[avatarIndex];
                    var bbox = faceData.BoundingBox;
                    
                    int targetWidth = Mathf.RoundToInt(bbox.width);
                    int targetHeight = Mathf.RoundToInt(bbox.height);
                    
                    if (_config.Version == "v15")
                    {
                        targetHeight += (int)_config.ExtraMargin; // extra_margin
                        // Clamp to original image height
                        targetHeight = Mathf.Min(targetHeight, faceData.OriginalTexture.height - Mathf.RoundToInt(bbox.y));
                    }
                    
                    // Resize decoded frame to face crop size
                    var resizedFrame = FrameUtils.ResizeFrame(rawDecodedTexture, targetWidth, targetHeight);
                    
                    // Convert face bbox to Vector4 format for blending (x1, y1, x2, y2)
                    var faceBbox = new Vector4(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);
                    string blendingMode = "jaw"; // version v15
                    // Use precomputed segmentation data for optimal performance
                    var blendedFrame = ImageBlendingHelper.BlendFaceWithOriginal(
                        faceData.OriginalTexture, 
                        resizedFrame, 
                        faceBbox,
                        faceData.CropBox,
                        faceData.BlurredMask, 
                        faceData.FaceLarge, 
                        _config.ExtraMargin,
                        blendingMode);
                    return blendedFrame;
                }
                return rawDecodedTexture;
            });
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Releases all resources used by the MuseTalkInference instance.
        /// Disposes of all ONNX models and prevents memory leaks.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases the unmanaged resources used by the MuseTalkInference and optionally releases the managed resources.
        /// </summary>
        /// <param name="disposing">True to release both managed and unmanaged resources; false to release only unmanaged resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources (ONNX models)
                    _unet?.Dispose();
                    _vaeEncoder?.Dispose();
                    _vaeDecoder?.Dispose();
                    _positionalEncoding?.Dispose();
                    _whisperModel?.Dispose();
                    _faceAnalysis?.Dispose();
                    
                    Logger.LogVerbose("[MuseTalkInference] All models disposed successfully");
                }
                
                // Release unmanaged resources and set large fields to null
                _reusableBatchArray = null;
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for the MuseTalkInference class.
        /// </summary>
        ~MuseTalkInference()
        {
            Dispose(false);
        }

        #endregion
    }
}
