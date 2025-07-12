using Microsoft.ML.OnnxRuntime;
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
    /// Face detection result
    /// </summary>
    internal class FaceDetectionResult
    {
        public Rect BoundingBox { get; set; }
        public Vector2[] Keypoints5 { get; set; }  // 5 keypoints from detection
        public Vector2[] Landmarks106 { get; set; }  // 106 landmarks
        public float DetectionScore { get; set; }
        
        public override string ToString()
        {
            return $"Face(bbox={BoundingBox}, conf={DetectionScore:F3})";
        }
    }

    /// <summary>
    /// Detection candidate struct for efficient processing (value type for better cache performance)
    /// </summary>
    internal struct DetectionCandidate
    {
        public float x1, y1, x2, y2;
        public float score;
        public unsafe fixed float keypoints[10]; // 5 keypoints * 2 coords
        
        public DetectionCandidate(float x1, float y1, float x2, float y2, float score)
        {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
            // keypoints will be initialized separately
        }
    }

    /// <summary>
    /// BiSeNet face parsing classes (19 total)
    /// </summary>
    internal enum FaceParsingClass
    {
        Background = 0,
        Skin = 1,           // Face region
        LeftBrow = 2,
        RightBrow = 3,
        LeftEye = 4,
        RightEye = 5,
        EyeGlass = 6,
        LeftEar = 7,
        RightEar = 8,
        Earring = 9,
        Nose = 10,
        Mouth = 11,         // Lips region
        UpperLip = 12,
        LowerLip = 13,
        Neck = 14,
        Necklace = 15,
        Cloth = 16,
        Hair = 17,
        Hat = 18
    }

    /// <summary>
    /// Consolidated face analysis class that handles detection, landmark extraction, and face parsing
    /// Uses the cleaner implementation from LivePortraitInference with added face parsing capabilities
    /// </summary>
    internal class FaceAnalysis : IDisposable
    {
        private static readonly string MODEL_RELATIVE_PATH_LIVE_PORTAIT = "LivePortrait";
        private static readonly string MODEL_RELATIVE_PATH_MUSE_TALK = "MuseTalk";
        // ONNX Models
        private readonly Model _detFace;      // Face detection model
        private readonly Model _landmark2d106; // 106 landmark detection model
        private readonly Model _landmarkRunner; // Landmark refinement model
        private readonly Model _faceParsing; // Face parsing/segmentation model
        
        // Configuration
        private readonly LiveTalkConfig _config;
        private bool _disposed = false;
        
        // Detection parameters
        private const float DetectionThreshold = 0.5f;
        private const float NmsThreshold = 0.4f;
        private const int InputSize = 512;
        private const int FeatMapCount = 3;
        private static readonly int[] FeatStrideFpn = { 8, 16, 32 };
        
        // Landmark parameters
        private const int LandmarkInputSize = 192;
        private const int LandmarkRunnerSize = 224;

        // Cache for anchor centers. Make sure to clear this cache after processing each frame.
        private readonly Dictionary<string, float[,]> _centerCache = new();
        private static FaceAnalysis _instance = null;

        public static FaceAnalysis CreateOrGetInstance(LiveTalkConfig config)
        {
            _instance ??= new FaceAnalysis(config);
            return _instance;
        }
        
        public bool IsInitialized { get; private set; }
        
        private FaceAnalysis(LiveTalkConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            
            try
            {
                // Load all required models
                _detFace = new Model(_config, "det_10g", MODEL_RELATIVE_PATH_LIVE_PORTAIT, ExecutionProvider.CoreML);
                _landmark2d106 = new Model(_config, "2d106det", MODEL_RELATIVE_PATH_LIVE_PORTAIT, ExecutionProvider.CoreML);
                _landmarkRunner = new Model(_config, "landmark", MODEL_RELATIVE_PATH_LIVE_PORTAIT, ExecutionProvider.CoreML);
                
                _faceParsing = new Model(_config, "face_parsing", MODEL_RELATIVE_PATH_MUSE_TALK, ExecutionProvider.CoreML);
                
                bool allInitialized = _detFace != null && _landmark2d106 != null && _landmarkRunner != null && _faceParsing != null;
                
                if (!allInitialized)
                {
                    var failedModels = new List<string>();
                    if (_detFace == null) failedModels.Add("DetFace");
                    if (_landmark2d106 == null) failedModels.Add("Landmark106");
                    if (_landmarkRunner == null) failedModels.Add("LandmarkRunner");
                    if (_faceParsing == null) failedModels.Add("FaceParsing");
                    
                    throw new InvalidOperationException($"Failed to initialize face analysis models: {string.Join(", ", failedModels)}");
                }
                
                IsInitialized = true;
                Logger.LogVerbose("[FaceAnalysis] Initialized successfully");
            }
            catch (Exception e)
            {
                Logger.LogError($"[FaceAnalysis] Failed to initialize: {e.Message}");
                IsInitialized = false;
                throw;
            }
        }
        
        #region Public Methods - Face Analysis

        /// <summary>
        /// Starts the face analysis session for the face analysis models.
        /// </summary>
        public async Task StartFaceAnalysisSession()
        {
            await _detFace.StartSession();
            await _landmark2d106.StartSession();
            await _landmarkRunner.StartSession();
        }

        /// <summary>
        /// Ends the face analysis session for the face analysis models.
        /// </summary>
        public void EndFaceAnalysisSession()
        {
            _detFace.EndSession();
            _landmark2d106.EndSession();
            _landmarkRunner.EndSession();
        }

        /// <summary>
        /// Starts the face parsing session for the face parsing model.
        /// </summary>
        public async Task StartFaceParsingSession()
        {
            await _faceParsing.StartSession();
        }

        /// <summary>
        /// Ends the face parsing session for the face parsing model.
        /// </summary>
        public void EndFaceParsingSession()
        {
            _faceParsing.EndSession();
        }

        /// <summary>
        /// Analyzes faces in the input image and returns detected faces with landmarks, sorted by area.
        /// This is the main entry point for comprehensive face analysis, combining detection and landmark extraction.
        /// </summary>
        /// <param name="frame">The input frame to analyze for faces</param>
        /// <param name="maxFaces">Maximum number of faces to return (-1 for all detected faces)</param>
        /// <returns>A task containing a list of detected faces with landmarks, sorted by area (largest first)</returns>
        public async Task<List<FaceDetectionResult>> AnalyzeFaces(Frame frame)
        {
            if (!IsInitialized)
                throw new InvalidOperationException("FaceAnalysis not initialized");
                
            if (frame.data == null || frame.width <= 0 || frame.height <= 0)
                throw new ArgumentException("Invalid image parameters");
            
            var faces = await DetectFaces(frame);
            var finalFaces = new List<FaceDetectionResult>();
            foreach (var face in faces)
            {
                var landmarks = await GetLandmarks(frame, face);
                face.Landmarks106 = landmarks;
                finalFaces.Add(face);
            }
            
            finalFaces.Sort((a, b) => 
            {
                float areaA = a.BoundingBox.width * a.BoundingBox.height;
                float areaB = b.BoundingBox.width * b.BoundingBox.height;
                return areaB.CompareTo(areaA); // Descending order
            });
            
            return finalFaces;
        }

        /// <summary>
        /// Asynchronously refines facial landmarks using the landmark runner model for improved accuracy.
        /// This method crops the face region based on existing landmarks, runs refinement inference, and returns enhanced landmark positions.
        /// </summary>
        /// <param name="img">The input image containing the face</param>
        /// <param name="lmk">The initial landmark points to refine</param>
        /// <returns>A task containing refined landmark points with improved positional accuracy</returns>
        public async Task<Vector2[]> LandmarkRunner(Frame img, Vector2[] lmk)
        {
            var cropSize = 224;
            var cropDct = GetCropInfo(img, lmk, cropSize, 1.5f, -0.1f);
            var imgCrop = cropDct.ImageCrop;
            var inputTensor = FrameUtils.FrameToTensor(imgCrop, 1.0f / 255.0f, 0.0f);
            var inputs = new List<Tensor<float>>
            {
                inputTensor
            };
            
            var results = await _landmarkRunner.Run(inputs);
            var outputs = results.ToArray();
            var outPts = outputs[2].AsTensor<float>().ToArray();
            
            var refinedLmk = new Vector2[outPts.Length / 2];
            for (int i = 0; i < refinedLmk.Length; i++)
            {
                refinedLmk[i] = new Vector2(outPts[i * 2] * cropSize, outPts[i * 2 + 1] * cropSize);
            }
            
            refinedLmk = TransformLandmarksWithMatrix(refinedLmk, cropDct.Transform);
            return refinedLmk;
        }

        /// <summary>
        /// Asynchronously extracts facial landmarks and bounding boxes using a hybrid SCRFD+106landmark approach.
        /// This method combines SCRFD face detection with 106-point landmark extraction to provide accurate face localization
        /// </summary>
        /// <param name="frames">The list of input frames to process for face detection and landmark extraction</param>
        /// <param name="bboxShift">Optional vertical shift to apply to bounding boxes for fine-tuning face region positioning</param>
        /// <returns>A task containing a tuple of bounding box coordinates (as Vector4 list) and the corresponding processed frames</returns>
        public async Task<(List<Vector4>, List<Frame>)> GetLandmarkAndBbox(List<Frame> frames, int bboxShift = 0)
        {
            var coordsList = new List<Vector4>();
            var framesList = new List<Frame>();
            var CoordPlaceholder = Vector4.zero; // Matching InsightFaceHelper.CoordPlaceholder
            
            Logger.LogVerbose($"[FaceAnalysis] Processing {frames.Count} images with hybrid SCRFD+106landmark approach");
            
            var averageRangeMinus = new List<float>();
            var averageRangePlus = new List<float>();
            
            for (int idx = 0; idx < frames.Count; idx++)
            {
                var frame = frames[idx];
                
                // Step 1: Detect faces using SCRFD (matching InsightFaceHelper exactly)
                var faces = await DetectFaces(frame);   

                if (faces.Count == 0)
                {
                    Logger.LogWarning($"[FaceAnalysis] No face detected in image {idx} ({frame.width}x{frame.height})");
                    coordsList.Add(CoordPlaceholder);
                    continue;
                }
                
                // Get the best detection
                var face = faces[0];
                var bbox = face.BoundingBox;
                var scrfdKps = face.Keypoints5;
                
                // Step 2: Extract 106 landmarks using face-aligned crop (matching InsightFaceHelper flow)
                Vector2[] landmarks106;
                Vector4 finalBbox;
                
                if (scrfdKps != null && scrfdKps.Length >= 5)
                {
                    // Use existing GetLandmarks method which already gives us 106 landmarks
                    landmarks106 = await GetLandmarks(frame, face);
                    if (landmarks106 != null && landmarks106.Length >= 106)
                    {
                        // Calculate final bbox using hybrid approach (adapted for 106 landmarks)
                        finalBbox = CalculateHybridBbox106(landmarks106, bbox, bboxShift);
                        
                        // Calculate range information (adapted for 106 landmarks)
                        var (rangeMinus, rangePlus) = CalculateLandmarkRanges106(landmarks106);
                        averageRangeMinus.Add(rangeMinus);
                        averageRangePlus.Add(rangePlus);
                    }
                    else
                    {
                        Logger.LogWarning($"[FaceAnalysis] Failed to extract 106 landmarks for image {idx}");
                        finalBbox = CreateFallbackBbox(bbox, scrfdKps);
                    }
                }
                else
                {
                    Logger.LogWarning($"[FaceAnalysis] No SCRFD keypoints for image {idx}, using detection bbox");
                    finalBbox = CreateFallbackBbox(bbox, null);
                }

                float width_check = finalBbox.z - finalBbox.x;
                float height_check = finalBbox.w - finalBbox.y;
                if (height_check <= 0 || width_check <= 0 || finalBbox.x < 0)
                {
                    Logger.LogWarning($"[FaceAnalysis] Invalid landmark bbox: [{finalBbox.x:F1}, {finalBbox.y:F1}, {finalBbox.z:F1}, {finalBbox.w:F1}], using SCRFD bbox");
                    coordsList.Add(CreateFallbackBbox(bbox, scrfdKps));
                }
                else
                {
                    coordsList.Add(finalBbox);
                }
                
                // Store the processed data
                framesList.Add(frame);
            }
            
            return (coordsList, framesList);
        }
        
        /// <summary>
        /// Asynchronously creates a face mask with morphological operations for smooth blending.
        /// This method generates a base segmentation mask and applies morphological operations (dilation, erosion, blur)
        /// to create smooth edges suitable for seamless face blending
        /// </summary>
        /// <param name="frame">The input face frame to create a mask for</param>
        /// <param name="mode">The segmentation mode: "jaw" applies full morphological processing, others use light smoothing</param>
        /// <returns>A task containing the processed face mask with smooth edges ready for blending operations</returns>
        public async Task<Frame> GenerateFaceSegmentationMask(Frame frame, string mode = "jaw")
        {
            var baseMaskFrame = await GenerateFaceSegmentationMaskInternal(frame, mode);
            if (baseMaskFrame.data == null) return new Frame(null, 0, 0);
            
            var smoothedMaskFrame = ApplyMorphologicalOperations(baseMaskFrame, mode);
            return smoothedMaskFrame;
        }

        /// <summary>
        /// Get crop info from the input frame using facial landmarks and transformation parameters.
        /// This method calculates an affine transformation to extract and standardize the face region.
        /// </summary>
        /// <param name="frame">The input frame containing the face to crop</param>
        /// <param name="lmk">The facial landmarks defining the face position and orientation</param>
        /// <param name="dsize">The target size for the cropped face region</param>
        /// <param name="scale">The scaling factor to apply during cropping</param>
        /// <param name="vyRatio">The vertical offset ratio for crop positioning</param>
        /// <returns>A CropInfo object containing the cropped image, transformation matrices, and transformed landmarks</returns>
        public CropInfo GetCropInfo(Frame frame, Vector2[] lmk, int dsize, float scale, float vyRatio)
        {
            var (MInv, _) = EstimateSimilarTransformFromPts(lmk, dsize, scale, 0f, vyRatio, true);
            
            var imgCrop = FrameUtils.AffineTransformFrame(frame, MInv, dsize, dsize);
            var ptCrop = MathUtils.TransformPts(lmk, MInv);
            
            var Mo2c = MathUtils.GetCropTransform(MInv);
            var Mc2o = Mo2c.inverse;
            
            var cropInfo = new CropInfo
            {
                ImageCrop = imgCrop,
                Transform = Mc2o,
                InverseTransform = Mo2c,
                LandmarksCrop = ptCrop
            };
            
            return cropInfo;
        }

        #endregion

        #region Private Methods - Face Detection and Landmark Extraction

        /// <summary>
        /// Asynchronously detects faces in the input frame using SCRFD (Sample and Computation Redistribution for Face Detection) model.
        /// This method resizes the input image, runs face detection inference, and processes the results with NMS filtering.
        /// </summary>
        /// <param name="frame">The input frame to detect faces in</param>
        /// <returns>A task containing a list of detected faces with bounding boxes, keypoints, and confidence scores</returns>
        private async Task<List<FaceDetectionResult>> DetectFaces(Frame frame)
        {
            int height = frame.height;
            int width = frame.width;
            
            float imRatio = (float)height / width;
            
            int newHeight, newWidth;
            if (imRatio > 1)
            {
                newHeight = InputSize;
                newWidth = Mathf.FloorToInt(newHeight / imRatio);
            }
            else
            {
                newWidth = InputSize;
                newHeight = Mathf.FloorToInt(newWidth * imRatio);
            }
            
            float detScale = (float)newHeight / height;
            var resizedImg = FrameUtils.ResizeFrame(frame, newWidth, newHeight, SamplingMode.Bilinear);
            var detImg = new Frame(new byte[InputSize * InputSize * 3], InputSize, InputSize);
            
            Array.Clear(detImg.data, 0, detImg.data.Length);
            
            unsafe
            {
                fixed (byte* srcPtrFixed = resizedImg.data)
                fixed (byte* dstPtrFixed = detImg.data)
                {
                    byte* srcPtrLocal = srcPtrFixed;
                    byte* dstPtrLocal = dstPtrFixed;
                    
                    Parallel.For(0, newHeight, y =>
                    {
                        byte* srcRowPtr = srcPtrLocal + y * newWidth * 3;
                        byte* dstRowPtr = dstPtrLocal + y * InputSize * 3;
                        
                        int rowBytes = newWidth * 3; // RGB24 = 3 bytes per pixel
                        Buffer.MemoryCopy(srcRowPtr, dstRowPtr, rowBytes, rowBytes);
                    });
                }
            }
            
            var inputTensor = FrameUtils.FrameToTensor(detImg, 0.0078125f, -0.99609375f);
            
            var inputs = new List<Tensor<float>> { inputTensor };
            using var results = await _detFace.RunDisposable(inputs);
            var outputs = results.ToArray();
            
            var faces = ProcessDetectionResults(outputs, detScale);
            return faces;
        }

        /// <summary>
        /// Processes raw ONNX detection outputs from the SCRFD model to create structured face detection results.
        /// This method applies feature pyramid network processing, anchor decoding, NMS filtering, and coordinate transformation.
        /// Uses unsafe memory operations for maximum performance when processing large numbers of detection candidates.
        /// </summary>
        /// <param name="outputs">The raw ONNX model outputs containing scores, bounding box predictions, and keypoint predictions</param>
        /// <param name="detScale">The scaling factor to transform detection coordinates back to original image space</param>
        /// <returns>A list of validated face detection results with bounding boxes, keypoints, and confidence scores</returns>
        private unsafe List<FaceDetectionResult> ProcessDetectionResults(NamedOnnxValue[] outputs, float detScale)
        {
            var validDetections = new List<DetectionCandidate>();
            
            // Process each stride level
            for (int idx = 0; idx < FeatMapCount; idx++)
            {
                var scores = outputs[idx].AsTensor<float>().ToArray();
                var bboxPreds = outputs[idx + FeatMapCount].AsTensor<float>().ToArray();
                var kpsPreds = outputs[idx + FeatMapCount * 2].AsTensor<float>().ToArray();
                
                int stride = FeatStrideFpn[idx];
                int height = InputSize / stride;
                int width = InputSize / stride;
                
                // Scale predictions by stride
                for (int i = 0; i < bboxPreds.Length; i++) bboxPreds[i] *= stride;
                for (int i = 0; i < kpsPreds.Length; i++) kpsPreds[i] *= stride;
                
                // Get anchor centers
                var anchorCenters = GetAnchorCenters(height, width, stride);
                
                // Find valid detections
                for (int i = 0; i < scores.Length; i++)
                {
                    if (scores[i] >= DetectionThreshold)
                    {
                        float centerX = anchorCenters[i, 0];
                        float centerY = anchorCenters[i, 1];
                        
                        float x1 = centerX - bboxPreds[i * 4 + 0];
                        float y1 = centerY - bboxPreds[i * 4 + 1];
                        float x2 = centerX + bboxPreds[i * 4 + 2];
                        float y2 = centerY + bboxPreds[i * 4 + 3];
                        
                        var candidate = new DetectionCandidate(x1, y1, x2, y2, scores[i]);
                        
                        // Copy keypoints (keypoints is already a fixed array in the struct)
                        unsafe
                        {
                            for (int k = 0; k < 10; k += 2)
                            {
                                candidate.keypoints[k] = centerX + kpsPreds[i * 10 + k];
                                candidate.keypoints[k + 1] = centerY + kpsPreds[i * 10 + k + 1];
                            }
                        }
                        
                        validDetections.Add(candidate);
                    }
                }
            }
            
            if (validDetections.Count == 0)
                return new List<FaceDetectionResult>();
            
            // Scale by detection scale
            float invDetScale = 1.0f / detScale;
            for (int i = 0; i < validDetections.Count; i++)
            {
                var candidate = validDetections[i];
                candidate.x1 *= invDetScale;
                candidate.y1 *= invDetScale;
                candidate.x2 *= invDetScale;
                candidate.y2 *= invDetScale;
                
                unsafe
                {
                    // keypoints is already a fixed array in the struct
                    for (int k = 0; k < 10; k++)
                    {
                        candidate.keypoints[k] *= invDetScale;
                    }
                }
                validDetections[i] = candidate;
            }
            
            // Sort by score and apply NMS
            validDetections.Sort((a, b) => b.score.CompareTo(a.score));
            var keepMask = new bool[validDetections.Count];
            ApplyNMS(validDetections, keepMask);
            
            // Convert to final results
            var faces = new List<FaceDetectionResult>();
            for (int i = 0; i < validDetections.Count; i++)
            {
                if (keepMask[i])
                {
                    var candidate = validDetections[i];
                    var face = new FaceDetectionResult
                    {
                        BoundingBox = new Rect(candidate.x1, candidate.y1, 
                            candidate.x2 - candidate.x1, candidate.y2 - candidate.y1),
                        DetectionScore = candidate.score,
                        Keypoints5 = new Vector2[5]
                    };
                    
                    unsafe
                    {
                        // keypoints is already a fixed array in the struct
                        for (int k = 0; k < 5; k++)
                        {
                            face.Keypoints5[k] = new Vector2(candidate.keypoints[k * 2], candidate.keypoints[k * 2 + 1]);
                        }
                    }
                    
                    faces.Add(face);
                }
            }
            _centerCache.Clear();            
            return faces;
        }
        
        /// <summary>
        /// Generates anchor center coordinates for the feature pyramid network at a given stride level.
        /// This method creates a grid of anchor points used for face detection, with caching for performance optimization.
        /// </summary>
        /// <param name="height">The height of the feature map</param>
        /// <param name="width">The width of the feature map</param>
        /// <param name="stride">The stride factor for this pyramid level</param>
        /// <returns>A 2D array containing anchor center coordinates [totalAnchors, 2] where each row is [x, y]</returns>
        private float[,] GetAnchorCenters(int height, int width, int stride)
        {
            string key = $"{height}_{width}_{stride}";
            
            if (_centerCache.TryGetValue(key, out var cached))
                return cached;
            
            const int numAnchors = 2;
            int totalAnchors = height * width * numAnchors;
            var centers = new float[totalAnchors, 2];
            
            int idx = 0;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float x = w * stride;
                    float y = h * stride;
                    
                    for (int a = 0; a < numAnchors; a++)
                    {
                        centers[idx, 0] = x;
                        centers[idx, 1] = y;
                        idx++;
                    }
                }
            }
            
            if (_centerCache.Count < 100)
                _centerCache[key] = centers;
                
            return centers;
        }
        
        /// <summary>
        /// Applies Non-Maximum Suppression (NMS) to filter overlapping face detections.
        /// This method removes duplicate detections by calculating IoU (Intersection over Union) and keeping only the highest confidence detections.
        /// </summary>
        /// <param name="candidates">The list of detection candidates sorted by confidence score</param>
        /// <param name="keepMask">Output boolean array indicating which candidates to keep (true) or suppress (false)</param>
        private void ApplyNMS(List<DetectionCandidate> candidates, bool[] keepMask)
        {
            for (int i = 0; i < candidates.Count; i++)
                keepMask[i] = true;
            
            for (int i = 0; i < candidates.Count; i++)
            {
                if (!keepMask[i]) continue;
                
                var candidateA = candidates[i];
                float areaA = (candidateA.x2 - candidateA.x1) * (candidateA.y2 - candidateA.y1);
                
                for (int j = i + 1; j < candidates.Count; j++)
                {
                    if (!keepMask[j]) continue;
                    
                    var candidateB = candidates[j];
                    
                    float intersectionX1 = Mathf.Max(candidateA.x1, candidateB.x1);
                    float intersectionY1 = Mathf.Max(candidateA.y1, candidateB.y1);
                    float intersectionX2 = Mathf.Min(candidateA.x2, candidateB.x2);
                    float intersectionY2 = Mathf.Min(candidateA.y2, candidateB.y2);
                    
                    if (intersectionX1 < intersectionX2 && intersectionY1 < intersectionY2)
                    {
                        float intersectionArea = (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1);
                        float areaB = (candidateB.x2 - candidateB.x1) * (candidateB.y2 - candidateB.y1);
                        float unionArea = areaA + areaB - intersectionArea;
                        
                        if (intersectionArea / unionArea >= NmsThreshold)
                        {
                            keepMask[j] = false;
                        }
                    }
                }
            }
        }
        
        /// <summary>
        /// Asynchronously extracts 106 facial landmarks from a detected face using the landmark detection model.
        /// This method crops and aligns the face region, runs landmark inference, and transforms the results back to the original coordinate system.
        /// </summary>
        /// <param name="frame">The input frame containing the face</param>
        /// <param name="face">The detected face with bounding box information</param>
        /// <returns>A task containing an array of 106 landmark points in the original image coordinates</returns>
        private async Task<Vector2[]> GetLandmarks(Frame frame, FaceDetectionResult face)
        {
            const int inputSize = 192;
            var bbox = face.BoundingBox;
            
            float w = bbox.width;
            float h = bbox.height;
            
            Vector2 center = new(bbox.x + w * 0.5f, bbox.y + h * 0.5f);
            float rotate = 0f;
            float scale = inputSize / (Mathf.Max(w, h) * 1.5f);
            var (alignedFrame, transformMatrix) = FaceAlign(frame, center, inputSize, scale, rotate);
            var inputTensor = FrameUtils.FrameToTensor(alignedFrame, 1.0f, 0.0f);
            
            var inputs = new List<Tensor<float>> { inputTensor };
            var results = await _landmark2d106.Run(inputs);
            var output = results.First().AsTensor<float>().ToArray();
            
            var landmarks = new Vector2[output.Length / 2];
            for (int i = 0; i < landmarks.Length; i++)
            {
                float x = output[i * 2] + 1f;
                float y = output[i * 2 + 1] + 1f;
                x *= inputSize / 2f;
                y *= inputSize / 2f;
                landmarks[i] = new Vector2(x, y);
            }
            
            var IM = transformMatrix.inverse;
            landmarks = MathUtils.TransPoints2D(landmarks, IM);
            
            return landmarks;
        }
        
        /// <summary>
        /// Aligns a face region by applying rotation, scaling, and translation transformations.
        /// This method creates a standardized face crop with consistent orientation and size for landmark detection.
        /// </summary>
        /// <param name="frame">The input frame containing the face</param>
        /// <param name="center">The center point of the face for alignment</param>
        /// <param name="outputSize">The target size for the aligned face crop</param>
        /// <param name="scale">The scaling factor to apply during alignment</param>
        /// <param name="rotate">The rotation angle in degrees for face alignment</param>
        /// <returns>A tuple containing the aligned face crop and the transformation matrix used</returns>
        private (Frame, Matrix4x4) FaceAlign(Frame frame, Vector2 center, int outputSize, float scale, float rotate)
        {
            float scaleRatio = scale;
            float rot = rotate * Mathf.Deg2Rad;
            
            float cosRot = Mathf.Cos(rot);
            float sinRot = Mathf.Sin(rot);
            float outputSizeHalf = outputSize * 0.5f;
            
            float m00 = scaleRatio * cosRot;
            float m01 = -scaleRatio * sinRot;
            float m02 = outputSizeHalf - center.x * m00 - center.y * m01;
            
            float m10 = scaleRatio * sinRot;
            float m11 = scaleRatio * cosRot;
            float m12 = outputSizeHalf - center.x * m10 - center.y * m11;

            float[,] M = new float[,] {
                { m00, m01, m02 },
                { m10, m11, m12 }
            };
            
            var cropped = FrameUtils.AffineTransformFrame(frame, M, outputSize, outputSize);

            var transform = new Matrix4x4
            {
                m00 = M[0, 0], m01 = M[0, 1], m02 = 0f, m03 = M[0, 2],
                m10 = M[1, 0], m11 = M[1, 1], m12 = 0f, m13 = M[1, 2],
                m20 = 0f, m21 = 0f, m22 = 1f, m23 = 0f,
                m30 = 0f, m31 = 0f, m32 = 0f, m33 = 1f
            };

            return (cropped, transform);
        }

        /// <summary>
        /// Calculates the similarity transformation matrices for cropping and inverse transformation from facial landmarks.
        /// This method computes both forward and inverse affine transformation matrices used for face alignment and paste-back operations.
        /// </summary>
        /// <param name="pts">The facial landmark points used to calculate transformation</param>
        /// <param name="dsize">The target size for the transformed image</param>
        /// <param name="scale">The scaling factor applied during landmark analysis</param>
        /// <param name="vxRatio">The horizontal offset ratio for transformation positioning</param>
        /// <param name="vyRatio">The vertical offset ratio for transformation positioning</param>
        /// <param name="flagDoRot">If true, includes rotation in the transformation; if false, only applies scaling and translation</param>
        /// <returns>A tuple containing the inverse transformation matrix (for cropping) and forward transformation matrix (for paste-back)</returns>
        private (float[,], float[,]) EstimateSimilarTransformFromPts(Vector2[] pts, int dsize, float scale, float vxRatio, float vyRatio, bool flagDoRot)
        {
            var (center, size, angle) = ParseRectFromLandmark(pts, scale, true, vxRatio, vyRatio, false);
            
            float s = dsize / size.x;
            Vector2 tgtCenter = new(dsize / 2f, dsize / 2f);
            
            float[,] MInv;
            
            if (flagDoRot)
            {
                float costheta = Mathf.Cos(angle);
                float sintheta = Mathf.Sin(angle);
                float cx = center.x, cy = center.y;
                float tcx = tgtCenter.x, tcy = tgtCenter.y;
                
                MInv = new float[,] {
                    { s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy) },
                    { -s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy) }
                };
            }
            else
            {
                MInv = new float[,] {
                    { s, 0, tgtCenter.x - s * center.x },
                    { 0, s, tgtCenter.y - s * center.y }
                };
            }
            
            var MInvH = new float[3, 3] {
                { MInv[0, 0], MInv[0, 1], MInv[0, 2] },
                { MInv[1, 0], MInv[1, 1], MInv[1, 2] },
                { 0f, 0f, 1f }
            };
            
            var M = MathUtils.InvertMatrix3x3(MInvH);
            var M2x3 = new float[,] {
                { M[0, 0], M[0, 1], M[0, 2] },
                { M[1, 0], M[1, 1], M[1, 2] }
            };

            return (MInv, M2x3);
        }
        
        /// <summary>
        /// Parses center position, size, and rotation angle from facial landmarks to define a bounding rectangle.
        /// This method calculates the optimal rectangular region that encompasses the face with proper orientation.
        /// </summary>
        /// <param name="pts">The facial landmark points to analyze</param>
        /// <param name="scale">The scaling factor to apply to the computed size</param>
        /// <param name="needSquare">If true, forces the rectangle to be square by using the maximum dimension</param>
        /// <param name="vxRatio">The horizontal offset ratio for center positioning</param>
        /// <param name="vyRatio">The vertical offset ratio for center positioning</param>
        /// <param name="useDegFlag">If true, returns angle in degrees; if false, returns angle in radians</param>
        /// <returns>A tuple containing the center position, size vector, and rotation angle of the face rectangle</returns>
        private (Vector2, Vector2, float) ParseRectFromLandmark(Vector2[] pts, float scale, bool needSquare, float vxRatio, float vyRatio, bool useDegFlag)
        {
            var pt2 = ParsePt2FromPtX(pts, true);
            
            Vector2 uy = pt2[1] - pt2[0];
            float l = uy.magnitude;
            
            if (l <= 1e-3f)
            {
                uy = new Vector2(0f, 1f);
            }
            else
            {
                uy /= l;
            }
            
            Vector2 ux = new(uy.y, -uy.x);
            
            float angle = Mathf.Acos(ux.x);
            if (ux.y < 0)
            {
                angle = -angle;
            }
            
            float[,] M = new float[,] { { ux.x, ux.y }, { uy.x, uy.y } };
            
            Vector2 center0 = Vector2.zero;
            for (int i = 0; i < pts.Length; i++)
            {
                center0 += pts[i];
            }
            center0 /= pts.Length;
            
            Vector2[] rpts = new Vector2[pts.Length];
            for (int i = 0; i < pts.Length; i++)
            {
                Vector2 centered = pts[i] - center0;
                rpts[i] = new Vector2(
                    centered.x * M[0, 0] + centered.y * M[1, 0],
                    centered.x * M[0, 1] + centered.y * M[1, 1]
                );
            }
            
            Vector2 ltPt = new(float.MaxValue, float.MaxValue);
            Vector2 rbPt = new(float.MinValue, float.MinValue);
            
            for (int i = 0; i < rpts.Length; i++)
            {
                if (rpts[i].x < ltPt.x) ltPt.x = rpts[i].x;
                if (rpts[i].y < ltPt.y) ltPt.y = rpts[i].y;
                if (rpts[i].x > rbPt.x) rbPt.x = rpts[i].x;
                if (rpts[i].y > rbPt.y) rbPt.y = rpts[i].y;
            }
            
            Vector2 center1 = (ltPt + rbPt) / 2f;
            Vector2 size = rbPt - ltPt;
            
            if (needSquare)
            {
                float m = Mathf.Max(size.x, size.y);
                size.x = m;
                size.y = m;
            }
            
            size *= scale;
            
            Vector2 center = center0 + ux * center1.x + uy * center1.y;
            center = center + ux * (vxRatio * size.x) + uy * (vyRatio * size.y);
            
            if (useDegFlag)
            {
                angle *= Mathf.Rad2Deg;
            }
            
            return (center, size, angle);
        }
        
         /// <summary>
        /// Parses two reference points from facial landmarks with optional perpendicular transformation.
        /// When not using lip points, applies a perpendicular rotation to create a proper coordinate system.
        /// </summary>
        /// <param name="pts">The facial landmark points to process</param>
        /// <param name="useLip">If true, uses lip-based reference points; if false, applies perpendicular transformation to eye points</param>
        /// <returns>An array of two transformed reference points for face alignment calculations</returns>
        private Vector2[] ParsePt2FromPtX(Vector2[] pts, bool useLip)
        {
            var pt2 = ParsePt2FromPt106(pts, useLip);
            
            if (!useLip)
            {
                Vector2 v = pt2[1] - pt2[0];
                pt2[1] = new Vector2(pt2[0].x - v.y, pt2[0].y + v.x);
            }
            
            return pt2;
        }
        
        /// <summary>
        /// Parses two key reference points from the 106 facial landmarks for transformation calculation.
        /// Can extract either eye-based or eye-lip based reference points depending on the useLip parameter.
        /// </summary>
        /// <param name="pt106">The array of 106 facial landmarks</param>
        /// <param name="useLip">If true, uses center eye and center lip points; if false, uses left and right eye points</param>
        /// <returns>An array of two reference points used for face transformation calculations</returns>
        private Vector2[] ParsePt2FromPt106(Vector2[] pt106, bool useLip)
        {
            Vector2 ptLeftEye = (pt106[33] + pt106[35] + pt106[40] + pt106[39]) / 4f;
            Vector2 ptRightEye = (pt106[87] + pt106[89] + pt106[94] + pt106[93]) / 4f;
            
            Vector2[] pt2;
            
            if (useLip)
            {
                Vector2 ptCenterEye = (ptLeftEye + ptRightEye) / 2f;
                Vector2 ptCenterLip = (pt106[52] + pt106[61]) / 2f;
                pt2 = new Vector2[] { ptCenterEye, ptCenterLip };
            }
            else
            {
                pt2 = new Vector2[] { ptLeftEye, ptRightEye };
            }
            
            return pt2;
        }
        
        /// <summary>
        /// Transforms an array of 2D landmarks using a 4x4 transformation matrix.
        /// This method applies the transformation matrix to each landmark point, converting 2D coordinates to 3D
        /// and then back to 2D for the final result.
        /// </summary>
        /// <param name="landmarks">The array of 2D landmark points to transform</param>
        /// <param name="transform">The 4x4 transformation matrix to apply to the landmarks</param>
        /// <returns>An array of transformed 2D landmark points</returns>
        private Vector2[] TransformLandmarksWithMatrix(Vector2[] landmarks, Matrix4x4 transform)
        {
            var result = new Vector2[landmarks.Length];
            for (int i = 0; i < landmarks.Length; i++)
            {
                var transformed = transform.MultiplyPoint3x4(new Vector3(landmarks[i].x, landmarks[i].y, 0));
                result[i] = new Vector2(transformed.x, transformed.y);
            }
            return result;
        }
        
        /// <summary>
        /// Calculates a hybrid bounding box using landmark center positioning with SCRFD-like dimensions.
        /// This method adapts the InsightFaceHelper.CalculateHybridBbox algorithm for 106 landmarks, providing
        /// landmark-accurate centering while maintaining proper face coverage for downstream processing.
        /// </summary>
        /// <param name="landmarks106">The array of 106 facial landmarks for precise face centering</param>
        /// <param name="originalBbox">The original SCRFD detection bounding box for size reference</param>
        /// <param name="bboxShift">Optional vertical shift to apply for fine-tuning face region positioning</param>
        /// <returns>A Vector4 representing the hybrid bounding box coordinates (x1, y1, x2, y2)</returns>
        private Vector4 CalculateHybridBbox106(Vector2[] landmarks106, Rect originalBbox, int bboxShift)
        {
            // Get landmark center and bounds
            // landmark_center_x = np.mean(face_land_mark[:, 0])
            // landmark_center_y = np.mean(face_land_mark[:, 1])
            Vector2 landmarkCenter = Vector2.zero;
            for (int i = 0; i < landmarks106.Length; i++)
            {
                landmarkCenter += landmarks106[i];
            }
            landmarkCenter /= landmarks106.Length;
            
            // Use SCRFD detection size as reference for proper face coverage
            // fx1, fy1, fx2, fy2 = original_bbox
            // scrfd_w = fx2 - fx1
            // scrfd_h = fy2 - fy1
            float scrfdWidth = originalBbox.width;
            float scrfdHeight = originalBbox.height;
            
            // Create bbox centered on landmarks but with SCRFD-like dimensions
            // This ensures we have enough face area for blending while being landmark-accurate
            float faceWidth = scrfdWidth * 0.9f;
            float faceHeight = scrfdHeight * 0.9f;
            
            // Center the bbox on landmark center
            float x1 = Mathf.Max(0, landmarkCenter.x - faceWidth * 0.5f);
            float y1 = Mathf.Max(0, landmarkCenter.y - faceHeight * 0.5f);
            float x2 = x1 + faceWidth;
            float y2 = y1 + faceHeight;
            
            // Apply bbox shift if specified (adapted for 106 landmarks)
            if (bboxShift != 0)
            {
                // For 106 landmarks, use nose tip (around landmark 66-68 area) as reference
                // This is the 106-landmark equivalent of the 68-landmark point 29
                if (landmarks106.Length > 66)
                {
                    Vector2 noseTipCoord = landmarks106[66]; // Approximate nose tip in 106-landmark system
                    float shiftedY = noseTipCoord.y + bboxShift;
                    float yOffset = shiftedY - noseTipCoord.y;
                    y1 += yOffset;
                    y2 += yOffset;
                }
            }
            
            return new Vector4(x1, y1, x2, y2);
        }
        
        /// <summary>
        /// Calculates landmark range information using nose area landmarks for spatial analysis.
        /// This method adapts the InsightFaceHelper.CalculateLandmarkRanges algorithm for 106 landmarks,
        /// providing vertical range measurements that can be used for face region sizing and positioning.
        /// </summary>
        /// <param name="landmarks106">The array of 106 facial landmarks to analyze</param>
        /// <returns>A tuple containing range measurements (rangeMinus, rangePlus) based on nose area landmark distances</returns>
        private (float rangeMinus, float rangePlus) CalculateLandmarkRanges106(Vector2[] landmarks106)
        {
            if (landmarks106.Length < 68)
                return (20f, 20f); // Default values
            
            // For 106 landmarks, use nose area landmarks for range calculation
            // Map to equivalent nose landmarks in 106-point system (approximate mapping)
            float rangeMinus = Mathf.Abs(landmarks106[67].y - landmarks106[66].y); // Nose area
            float rangePlus = Mathf.Abs(landmarks106[66].y - landmarks106[65].y);  // Nose area
            
            return (rangeMinus, rangePlus);
        }
        
        /// <summary>
        /// Creates a fallback bounding box when landmark extraction fails.
        /// This method expands the original SCRFD detection bounding box by a small factor to ensure adequate face coverage,
        /// matching the InsightFaceHelper.CreateFallbackBbox algorithm exactly for consistency.
        /// </summary>
        /// <param name="originalBbox">The original face detection bounding box to expand</param>
        /// <param name="scrfdKps">The SCRFD keypoints (may be null if detection failed)</param>
        /// <returns>A Vector4 representing the expanded fallback bounding box coordinates (x1, y1, x2, y2)</returns>
        private Vector4 CreateFallbackBbox(Rect originalBbox, Vector2[] scrfdKps)
        {
            // fx1, fy1, fx2, fy2 = original_bbox
            // expansion_factor = 1.05
            // center_x, center_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            // scrfd_w, scrfd_h = fx2 - fx1, fy2 - fy1
            // new_w, new_h = scrfd_w * expansion_factor, scrfd_h * expansion_factor
            float expansionFactor = 1.05f;
            float centerX = originalBbox.x + originalBbox.width * 0.5f;
            float centerY = originalBbox.y + originalBbox.height * 0.5f;
            float scrfdW = originalBbox.width;
            float scrfdH = originalBbox.height;
            float newW = scrfdW * expansionFactor;
            float newH = scrfdH * expansionFactor;
            float x1 = Mathf.Max(0, centerX - newW * 0.5f);
            float y1 = Mathf.Max(0, centerY - newH * 0.5f);
            float x2 = x1 + newW;
            float y2 = y1 + newH;
            
            return new Vector4(x1, y1, x2, y2);
        }

        #endregion

        #region Private Methods - Face Segmentation

        /// <summary>
        /// Asynchronously generates a face segmentation mask using the BiSeNet model with optimized byte array processing.
        /// This method preprocesses the input image, runs BiSeNet inference to classify facial regions,
        /// and creates a binary mask based on the specified mode (jaw, neck, or raw).
        /// </summary>
        /// <param name="frame">The input face frame to generate a segmentation mask for</param>
        /// <param name="mode">The segmentation mode: "jaw" for face+mouth, "neck" for face+mouth+neck, "raw" for face+lips only</param>
        /// <returns>A task containing the generated segmentation mask as a binary frame (white=face region, black=background)</returns>
        private async Task<Frame> GenerateFaceSegmentationMaskInternal(Frame frame, string mode = "jaw")
        {
            if (!IsInitialized)
            {
                Logger.LogError("[FaceAnalysis] Face analysis not initialized");
                return new Frame(null, 0, 0);
            }
            
            try
            {
                // Step 1: Preprocess image for BiSeNet (512x512, normalized) directly from byte array
                var preprocessedTensor = PreprocessImageForBiSeNet(frame);
                
                // Step 2: Run ONNX inference
                var parsingResult = await RunBiSeNetInference(preprocessedTensor);
                
                // Step 3: Post-process to create mask based on mode, returning byte array
                var maskFrame = PostProcessParsingResult(parsingResult, mode, frame.width, frame.height);
                
                return maskFrame;
            }
            catch (Exception e)
            {
                Logger.LogError($"[FaceAnalysis] Face parsing failed: {e.Message}");
                return new Frame(null, 0, 0);
            }
        }
        
        /// <summary>
        /// Preprocesses an input image for BiSeNet inference by resizing and applying ImageNet normalization.
        /// This method resizes the image to 512x512 and applies per-channel normalization using ImageNet mean and standard deviation values.
        /// </summary>
        /// <param name="inputImage">The input image frame to preprocess</param>
        /// <returns>A normalized tensor ready for BiSeNet model inference with shape [1, 3, 512, 512]</returns>
        private DenseTensor<float> PreprocessImageForBiSeNet(Frame inputImage)
        {
            // Resize to BiSeNet input size (512x512) - now uses optimized ResizeTextureToExactSize with byte arrays
            var resizedImageData = FrameUtils.ResizeFrame(inputImage, 512, 512, SamplingMode.Bilinear);
            
            // ImageNet normalization: (pixel/255 - mean) / std for each channel
            // R: (pixel/255 - 0.485) / 0.229, G: (pixel/255 - 0.456) / 0.224, B: (pixel/255 - 0.406) / 0.225
            // Transform to: pixel * (1/255/std) + (-mean/std)
            var multipliers = new float[] 
            { 
                1.0f / (255.0f * 0.229f),  // R: 1/(255*0.229) 
                1.0f / (255.0f * 0.224f),  // G: 1/(255*0.224)
                1.0f / (255.0f * 0.225f)   // B: 1/(255*0.225)
            };
            var offsets = new float[] 
            { 
                -0.485f / 0.229f,  // R: -mean_r/std_r
                -0.456f / 0.224f,  // G: -mean_g/std_g  
                -0.406f / 0.225f   // B: -mean_b/std_b
            };
            
            return FrameUtils.FrameToTensor(resizedImageData, multipliers, offsets);
        }

        /// <summary>
        /// Post-processes BiSeNet model output to create a 2D segmentation map using optimized parallel processing.
        /// This method applies argmax operation across 19 face parsing classes to determine the most likely class for each pixel,
        /// using unsafe memory operations and parallel processing for maximum performance.
        /// </summary>
        /// <param name="output">The raw BiSeNet model output tensor with shape [1, 19, 512, 512]</param>
        /// <returns>A 2D integer array [512, 512] where each value represents the face parsing class ID (0-18)</returns>
        private unsafe int[,] PostProcessParsingResult(Tensor<float> output)
        {
            // Convert output to segmentation map [512, 512]
            // Output shape: [1, 19, 512, 512] - 19 face parsing classes
            var parsingMap = new int[512, 512];
            
            // Get tensor data array - unfortunately ONNX tensors don't expose direct memory access
            var outputArray = output.ToArray();
            
            // Pre-calculate tensor strides for efficient pointer arithmetic
            // Tensor layout: [batch=1, classes=19, height=512, width=512]
            const int imageSize = 512 * 512;
            const int classStride = imageSize; // Elements per class channel
            
            // OPTIMIZED: Get unsafe pointer to array data outside parallel loop
            fixed (float* outputPtr = outputArray)
            {
                // Convert pointer to IntPtr to pass into parallel lambda (C# limitation workaround)
                IntPtr outputPtrAddr = new(outputPtr);
                
                // OPTIMIZED: Maximum parallelism across all 512×512 pixels (262,144-way parallelism)
                // Apply argmax operation with direct unsafe memory access and stride-based calculation
                System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
                {
                    // Convert IntPtr back to unsafe pointer inside lambda
                    float* unsafeOutputPtr = (float*)outputPtrAddr.ToPointer();
                    
                    // Calculate x, y coordinates from linear pixel index using bit operations
                    int y = pixelIndex >> 9;  // Divide by 512 (right shift 9 bits: 2^9 = 512)
                    int x = pixelIndex & 511; // Modulo 512 (bitwise AND with 511: 2^9-1 = 511)
                    
                    // Find class with maximum probability using direct unsafe memory access
                    int maxClass = 0;
                    float* pixelPtr = unsafeOutputPtr + pixelIndex; // Pointer to class 0 for this pixel
                    float maxProb = *pixelPtr; // Dereference pointer - no bounds checking!
                    
                    // OPTIMIZED: Direct pointer arithmetic for argmax (19 classes total)
                    // Check classes 1-18 using pointer arithmetic - fastest possible access
                    for (int c = 1; c < 19; c++)
                    {
                        float* classPtr = pixelPtr + c * classStride; // Pointer to class c for this pixel
                        float prob = *classPtr; // Direct memory access - no bounds checking!
                        if (prob > maxProb)
                        {
                            maxProb = prob;
                            maxClass = c;
                        }
                    }
                    
                    // Store result in parsing map
                    parsingMap[y, x] = maxClass;
                });
            }
            
            return parsingMap;
        }
        
        /// <summary>
        /// Asynchronously runs BiSeNet model inference to generate face parsing results.
        /// This method executes the preprocessed tensor through the BiSeNet ONNX model and post-processes the output
        /// to create a structured segmentation map for face region classification.
        /// </summary>
        /// <param name="inputTensor">The preprocessed input tensor ready for BiSeNet inference</param>
        /// <returns>A task containing a 2D segmentation map with face parsing class assignments</returns>
        private async Task<int[,]> RunBiSeNetInference(DenseTensor<float> inputTensor)
        {
            // Create input for ONNX model
            var inputs = new List<Tensor<float>>
            {
                inputTensor
            };
            
            // Run inference using ModelUtils for consistency
            var results = await _faceParsing.Run(inputs);
            var output = results.First().AsTensor<float>();
            
            return PostProcessParsingResult(output);
        }
        
        /// <summary>
        /// Post-processes the segmentation map to create a binary mask frame based on the specified mode.
        /// This method uses optimized parallel processing with unsafe memory operations to convert class-based 
        /// segmentation into binary masks for different face regions (jaw, neck, or raw mode).
        /// </summary>
        /// <param name="parsingMap">The 2D segmentation map with face parsing class assignments</param>
        /// <param name="mode">The mask mode: "jaw" for face+mouth, "neck" for face+mouth+neck, "raw" for face+lips</param>
        /// <param name="targetWidth">The target width for the output mask frame</param>
        /// <param name="targetHeight">The target height for the output mask frame</param>
        /// <returns>A binary mask frame where white pixels represent the selected face regions and black pixels represent background</returns>
        private unsafe Frame PostProcessParsingResult(int[,] parsingMap, string mode, int targetWidth, int targetHeight)
        {
            // Create mask data directly as byte array (RGB24: 3 bytes per pixel)
            const int maskWidth = 512;
            const int maskHeight = 512;
            int totalBytes = maskWidth * maskHeight * 3;
            var maskFrame = new Frame(new byte[totalBytes], maskWidth, maskHeight);
            
            // Pre-calculate class IDs for each mode to avoid string comparison in hot path
            bool* classLookup = stackalloc bool[19]; // 19 face parsing classes
            
            string lowerMode = mode.ToLower();
            if (lowerMode == "neck")
            {
                // Include face, lips, and neck regions
                classLookup[(int)FaceParsingClass.Skin] = true;
                classLookup[(int)FaceParsingClass.Mouth] = true;
                classLookup[(int)FaceParsingClass.UpperLip] = true;
                classLookup[(int)FaceParsingClass.LowerLip] = true;
                classLookup[(int)FaceParsingClass.Neck] = true;
            }
            else if (lowerMode == "jaw")
            {
                // Include face and mouth regions (for talking head)
                classLookup[(int)FaceParsingClass.Skin] = true;
                classLookup[(int)FaceParsingClass.Mouth] = true;
                classLookup[(int)FaceParsingClass.UpperLip] = true;
                classLookup[(int)FaceParsingClass.LowerLip] = true;
            }
            else // "raw" or default
            {
                // Include face and lip regions
                classLookup[(int)FaceParsingClass.Skin] = true;
                classLookup[(int)FaceParsingClass.Mouth] = true;
                classLookup[(int)FaceParsingClass.UpperLip] = true;
                classLookup[(int)FaceParsingClass.LowerLip] = true;
            }
            
            // OPTIMIZED: Maximum parallelism across all 512×512 pixels (262,144-way parallelism)
            // Apply mode-specific processing with stride-based coordinate calculation
            const int imageSize = 512 * 512;
            
            fixed (byte* maskPtrFixed = maskFrame.data)
            {
                // Capture pointer in local variable to avoid lambda closure issues
                byte* maskPtrLocal = maskPtrFixed;
                
                System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
                {
                    // Calculate x, y coordinates from linear pixel index using bit operations
                    int y = pixelIndex >> 9;  // Divide by 512 (right shift 9 bits: 2^9 = 512)
                    int x = pixelIndex & 511; // Modulo 512 (bitwise AND with 511: 2^9-1 = 511)
                    
                    // Get class ID from parsing map
                    int classId = parsingMap[y, x];
                    
                    // Fast lookup using pre-calculated boolean array (no switch statement)
                    bool isForeground = classId < 19 && classLookup[classId];
                    
                    // Calculate target pixel pointer using stride arithmetic (no Y-flipping needed for byte arrays)
                    byte* pixelPtr = maskPtrLocal + ((y << 9) + x) * 3; // (y * 512 + x) * 3 for RGB24
                    
                    // Set mask value directly in memory (RGB24: all channels same for grayscale)
                    byte maskValue = isForeground ? (byte)255 : (byte)0;
                    pixelPtr[0] = maskValue; // R
                    pixelPtr[1] = maskValue; // G  
                    pixelPtr[2] = maskValue; // B
                });
            }
            
            // Resize to target dimensions if needed using optimized resize
            if (targetWidth != 512 || targetHeight != 512)
            {
                var resizedMaskFrame = FrameUtils.ResizeFrame(maskFrame, targetWidth, targetHeight, SamplingMode.Bilinear);
                return resizedMaskFrame;
            }
            
            return maskFrame;
        }
        
        /// <summary>
        /// Applies morphological operations to smooth and refine face mask edges.
        /// This method uses dilation, erosion, and Gaussian blur operations to create smooth mask boundaries
        /// suitable for seamless face blending, with different processing intensity based on the mode.
        /// </summary>
        /// <param name="frame">The input binary mask frame to process</param>
        /// <param name="mode">The processing mode: "jaw" applies full morphological processing, others use light smoothing</param>
        /// <returns>A processed mask frame with smooth edges optimized for blending operations</returns>
        private unsafe Frame ApplyMorphologicalOperations(Frame frame, string mode)
        {
            if (mode.ToLower() == "jaw")
            {
                // Apply morphological operations using unsafe pointers
                var dilatedFrame = FrameUtils.ApplyDilation(frame, 3);
                var erodedFrame = FrameUtils.ApplyErosion(dilatedFrame, 2);
                
                // Apply optimized Gaussian blur
                return FrameUtils.ApplySimpleGaussianBlur(erodedFrame, 5);
            }
            else
            {
                // For other modes, just apply light smoothing
                return FrameUtils.ApplySimpleGaussianBlur(frame, 3);
            }
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Releases all resources used by the FaceAnalysis instance.
        /// Disposes of all ONNX models and prevents memory leaks.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases the unmanaged resources used by the FaceAnalysis and optionally releases the managed resources.
        /// </summary>
        /// <param name="disposing">True to release both managed and unmanaged resources; false to release only unmanaged resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources (ONNX models)
                    _detFace?.Dispose();
                    _landmark2d106?.Dispose();
                    _landmarkRunner?.Dispose();
                    _faceParsing?.Dispose();
                    
                    Logger.LogVerbose("[FaceAnalysis] All models disposed successfully");
                }
                
                // Clear cache and set disposed flag
                _centerCache?.Clear();
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for the FaceAnalysis class.
        /// </summary>
        ~FaceAnalysis()
        {
            Dispose(false);
        }

        #endregion
    }
}
