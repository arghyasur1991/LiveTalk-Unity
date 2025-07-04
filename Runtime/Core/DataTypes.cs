using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace LiveTalk.Core
{
    #region Enums

    /// <summary>
    /// Specifies the execution provider for ONNX model inference.
    /// Different providers offer varying performance characteristics and hardware acceleration capabilities.
    /// </summary>
    internal enum ExecutionProvider
    {
        /// <summary>
        /// CPU execution provider - universal compatibility, moderate performance
        /// </summary>
        CPU,
        
        /// <summary>
        /// CUDA execution provider - GPU acceleration for NVIDIA cards, high performance
        /// </summary>
        CUDA,
        
        /// <summary>
        /// CoreML execution provider - Apple Silicon/macOS acceleration, optimized for Apple hardware
        /// </summary>
        CoreML
    }

    /// <summary>
    /// Specifies the numerical precision for ONNX model weights and computations.
    /// Higher precision offers better accuracy while lower precision provides better performance and memory usage.
    /// </summary>
    internal enum Precision
    {
        /// <summary>
        /// 32-bit floating point - highest accuracy, largest memory footprint
        /// </summary>
        FP32,
        
        /// <summary>
        /// 16-bit floating point - balanced accuracy and performance, reduced memory usage
        /// </summary>
        FP16,
        
        /// <summary>
        /// 8-bit integer - fastest inference, smallest memory footprint, potential accuracy loss
        /// </summary>
        INT8
    }

    /// <summary>
    /// Specifies the sampling mode for texture resizing operations.
    /// Different modes provide trade-offs between quality and performance for various use cases.
    /// </summary>
    internal enum SamplingMode
    {
        /// <summary>
        /// Bilinear interpolation - higher quality, slower performance (default for ML preprocessing)
        /// </summary>
        Bilinear,
        
        /// <summary>
        /// Point/Nearest neighbor sampling - faster performance, lower quality (good for face detection)
        /// </summary>
        Point
    }

    /// <summary>
    /// Specifies the type of morphological operation for image processing.
    /// Used for mask refinement and edge processing in face segmentation.
    /// </summary>
    internal enum MorphologyOperation
    {
        /// <summary>
        /// Dilation operation - expands bright regions by finding maximum values in kernel neighborhood
        /// </summary>
        Dilation,
        
        /// <summary>
        /// Erosion operation - shrinks bright regions by finding minimum values in kernel neighborhood
        /// </summary>
        Erosion
    }

    /// <summary>
    /// Specifies the direction for blur operations in separable blur algorithms.
    /// Allows for efficient two-pass blur implementation with horizontal and vertical passes.
    /// </summary>
    internal enum BlurDirection
    {
        /// <summary>
        /// Horizontal blur pass - samples pixels along the X-axis
        /// </summary>
        Horizontal,
        
        /// <summary>
        /// Vertical blur pass - samples pixels along the Y-axis
        /// </summary>
        Vertical
    }

    #endregion

    #region Configuration Classes

    /// <summary>
    /// Configuration class for ONNX model initialization and execution parameters.
    /// Encapsulates model identification, file paths, and hardware acceleration preferences.
    /// </summary>
    internal class ModelConfig
    {
        /// <summary>
        /// The name identifier for this model
        /// </summary>
        public string modelName = "";
        
        /// <summary>
        /// The relative path to the model file from the base model directory
        /// </summary>
        public string modelRelativePath = "";
        
        /// <summary>
        /// The preferred execution provider for hardware acceleration
        /// </summary>
        public ExecutionProvider preferredExecutionProvider = ExecutionProvider.CPU;
        
        /// <summary>
        /// The numerical precision for model weights and computations
        /// </summary>
        public Precision precision = Precision.FP32;

        /// <summary>
        /// Initializes a new instance of the ModelConfig class with the specified parameters.
        /// </summary>
        /// <param name="modelName">The name identifier for this model</param>
        /// <param name="modelRelativePath">The relative path to the model file</param>
        /// <param name="preferredExecutionProvider">The preferred execution provider for hardware acceleration</param>
        /// <param name="precision">The numerical precision for model operations</param>
        public ModelConfig(string modelName, string modelRelativePath, ExecutionProvider preferredExecutionProvider, Precision precision)
        {
            this.modelName = modelName;
            this.modelRelativePath = modelRelativePath;
            this.preferredExecutionProvider = preferredExecutionProvider;            
            this.precision = precision;
        }
    }

    #endregion

    #region Core Data Structures

    /// <summary>
    /// Represents a frame of image data in RGB24 format with width and height dimensions.
    /// This structure is used throughout the LiveTalk pipeline for efficient image processing
    /// and provides a unified interface for raw byte array image data.
    /// </summary>
    internal struct Frame
    {
        /// <summary>
        /// The raw image data in RGB24 format (3 bytes per pixel: R, G, B)
        /// </summary>
        public byte[] data;
        
        /// <summary>
        /// The width of the image in pixels
        /// </summary>
        public int width;
        
        /// <summary>
        /// The height of the image in pixels
        /// </summary>
        public int height;

        /// <summary>
        /// Initializes a new Frame with the specified image data and dimensions.
        /// </summary>
        /// <param name="data">The raw image data in RGB24 format</param>
        /// <param name="width">The width of the image in pixels</param>
        /// <param name="height">The height of the image in pixels</param>
        public Frame(byte[] data, int width, int height)
        {
            this.data = data;
            this.width = width;
            this.height = height;
        }        
    }

    /// <summary>
    /// RGB24 pixel structure for efficient 3-byte color operations.
    /// Uses sequential memory layout with tight packing for optimal performance in unsafe operations.
    /// Currently unused but available for future optimizations requiring direct pixel manipulation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    internal struct RGB24
    {
        /// <summary>
        /// Red color component (0-255)
        /// </summary>
        public byte r;
        
        /// <summary>
        /// Green color component (0-255)
        /// </summary>
        public byte g;
        
        /// <summary>
        /// Blue color component (0-255)
        /// </summary>
        public byte b;
        
        /// <summary>
        /// Initializes a new RGB24 pixel with the specified color components.
        /// </summary>
        /// <param name="r">Red component (0-255)</param>
        /// <param name="g">Green component (0-255)</param>
        /// <param name="b">Blue component (0-255)</param>
        public RGB24(byte r, byte g, byte b)
        {
            this.r = r;
            this.g = g;
            this.b = b;
        }
    }

    #endregion

    #region Face Analysis Data Structures

    /// <summary>
    /// Comprehensive face detection and landmark data with optimized memory usage.
    /// Uses Frame structures (byte arrays) for internal storage instead of Texture2D objects
    /// to provide better memory efficiency and performance in the inference pipeline.
    /// </summary>
    internal class FaceData
    {
        /// <summary>
        /// Indicates whether a face was successfully detected and processed
        /// </summary>
        public bool HasFace { get; set; }
        
        /// <summary>
        /// The bounding box coordinates of the detected face in the original image
        /// </summary>
        public Rect BoundingBox { get; set; }
        
        /// <summary>
        /// The facial landmark points for this face (typically 106 points)
        /// </summary>
        public Vector2[] Landmarks { get; set; }
        
        /// <summary>
        /// The cropped face texture data in RGB24 format, standardized for processing
        /// </summary>
        public Frame CroppedFaceTexture { get; set; }
        
        /// <summary>
        /// The original full image containing this face, stored as RGB24 frame data
        /// </summary>
        public Frame OriginalTexture { get; set; }
        
        /// <summary>
        /// Face parsing mask for region-specific processing (if face parsing is enabled)
        /// </summary>
        public Frame FaceMask { get; set; }
        
        /// <summary>
        /// Expanded face region crop used for segmentation processing
        /// Computed once during avatar processing and cached for performance
        /// </summary>
        public Frame FaceLarge { get; set; }
        
        /// <summary>
        /// BiSeNet-generated segmentation mask for face region identification
        /// Precomputed during avatar processing for efficient frame blending
        /// </summary>
        public Frame SegmentationMask { get; set; }
        
        /// <summary>
        /// Face bounding box coordinates with version-specific adjustments applied
        /// </summary>
        public Vector4 AdjustedFaceBbox { get; set; }
        
        /// <summary>
        /// Expanded crop box coordinates used for face region processing
        /// </summary>
        public Vector4 CropBox { get; set; }
        
        /// <summary>
        /// Small mask cropped to the face region dimensions
        /// Precomputed for optimal blending performance during frame generation
        /// </summary>
        public Frame MaskSmall { get; set; }
        
        /// <summary>
        /// Full-size mask with the small mask pasted back to original dimensions
        /// Used for seamless blending of generated faces with original images
        /// </summary>
        public Frame FullMask { get; set; }
        
        /// <summary>
        /// Mask with upper boundary ratio applied to preserve natural face transitions
        /// Ensures smooth blending at the face-background boundary
        /// </summary>
        public Frame BoundaryMask { get; set; }
        
        /// <summary>
        /// Final blurred mask optimized for smooth face blending operations
        /// Applies Gaussian blur to eliminate harsh edges and artifacts
        /// </summary>
        public Frame BlurredMask { get; set; }
    }
    
    /// <summary>
    /// Processed avatar data container holding face regions and their corresponding latent representations.
    /// This class stores the complete processed avatar information needed for video generation,
    /// including detected faces and their VAE-encoded latent vectors.
    /// </summary>
    internal class AvatarData
    {
        /// <summary>
        /// List of detected and processed face regions from the avatar images
        /// Each FaceData contains complete face information including crops, masks, and metadata
        /// </summary>
        public List<FaceData> FaceRegions { get; set; } = new List<FaceData>();
        
        /// <summary>
        /// List of VAE-encoded latent representations corresponding to each face region
        /// These latents are used as input to the UNet model for frame generation
        /// </summary>
        public List<float[]> Latents { get; set; } = new List<float[]>();
    }
    
    /// <summary>
    /// Precomputed segmentation data container for efficient frame blending operations.
    /// This class stores all segmentation-related data computed once during avatar processing
    /// to optimize performance during real-time frame generation.
    /// </summary>
    internal class SegmentationData
    {
        /// <summary>
        /// Expanded face region crop used for segmentation processing
        /// </summary>
        public Frame FaceLarge { get; set; }
        
        /// <summary>
        /// BiSeNet-generated segmentation mask for face region identification
        /// </summary>
        public Frame SegmentationMask { get; set; }
        
        /// <summary>
        /// Face bounding box coordinates with version-specific adjustments applied
        /// </summary>
        public Vector4 AdjustedFaceBbox { get; set; }
        
        /// <summary>
        /// Expanded crop box coordinates used for face region processing
        /// </summary>
        public Vector4 CropBox { get; set; }
        
        /// <summary>
        /// Small mask cropped to the face region dimensions for efficient processing
        /// </summary>
        public Frame MaskSmall { get; set; }
        
        /// <summary>
        /// Full-size mask with the small mask pasted back to original dimensions
        /// </summary>
        public Frame FullMask { get; set; }
        
        /// <summary>
        /// Mask with upper boundary ratio applied to preserve natural face transitions
        /// </summary>
        public Frame BoundaryMask { get; set; }
        
        /// <summary>
        /// Final blurred mask optimized for smooth face blending operations
        /// </summary>
        public Frame BlurredMask { get; set; }
    }

    #endregion

    #region Audio Processing Data Structures

    /// <summary>
    /// Audio feature data extracted from Whisper model for temporal synchronization.
    /// Each chunk contains flattened features with dimensions [time_steps × layers × features] = [10 × 5 × 384] = [19200],
    /// </summary>
    internal class AudioFeatures
    {
        /// <summary>
        /// List of audio feature chunks, each containing flattened temporal features from Whisper
        /// </summary>
        public List<float[]> FeatureChunks { get; set; } = new List<float[]>();
        
        /// <summary>
        /// The sample rate of the original audio data in Hz
        /// </summary>
        public int SampleRate { get; set; }
        
        /// <summary>
        /// The total duration of the audio in seconds
        /// </summary>
        public float Duration { get; set; }
        
        /// <summary>
        /// The number of feature chunks, equivalent to the number of output frames
        /// </summary>
        public int ChunkCount => FeatureChunks.Count;
    }

    #endregion

    #region Motion Analysis Data Structures

    /// <summary>
    /// Crop information container for face processing.
    /// Contains cropped images at different resolutions, transformed landmarks, and transformation matrices
    /// for bidirectional coordinate transformation between original and cropped spaces.
    /// </summary>
    internal class CropInfo
    {
        /// <summary>
        /// The cropped face image at original crop resolution (typically 512x512)
        /// </summary>
        public Frame ImageCrop { get; set; }
        
        /// <summary>
        /// The cropped face image resized to 256x256 for neural network processing
        /// </summary>
        public Frame ImageCrop256x256 { get; set; }
        
        /// <summary>
        /// Facial landmarks transformed to the crop coordinate system
        /// </summary>
        public Vector2[] LandmarksCrop { get; set; }
        
        /// <summary>
        /// Facial landmarks transformed to the 256x256 crop coordinate system
        /// </summary>
        public Vector2[] LandmarksCrop256x256 { get; set; }
        
        /// <summary>
        /// Transformation matrix from crop coordinates to original image coordinates
        /// </summary>
        public Matrix4x4 Transform { get; set; }
        
        /// <summary>
        /// Inverse transformation matrix from original image coordinates to crop coordinates
        /// </summary>
        public Matrix4x4 InverseTransform { get; set; }
    }
    
    /// <summary>
    /// Complete result container for source image processing in LivePortrait inference.
    /// Contains all processed data needed for driving frame animation, including crops,
    /// features, motion parameters, and transformation matrices.
    /// </summary>
    internal class ProcessSourceImageResult
    {
        /// <summary>
        /// Crop information with images and transformation matrices
        /// </summary>
        public CropInfo CropInfo { get; set; }
        
        /// <summary>
        /// Preprocessed source image with standardized dimensions
        /// </summary>
        public Frame SrcImg { get; set; }
        
        /// <summary>
        /// Original paste-back mask for seamless face compositing
        /// </summary>
        public Frame MaskOri { get; set; }
        
        /// <summary>
        /// Extracted motion information from the source image
        /// </summary>
        public MotionInfo XsInfo { get; set; }
        
        /// <summary>
        /// Rotation matrix derived from source image motion parameters
        /// </summary>
        public float[,] Rs { get; set; }
        
        /// <summary>
        /// 3D appearance features extracted from the source image
        /// </summary>
        public Tensor<float> Fs { get; set; }
        
        /// <summary>
        /// Transformed keypoints from the source image in canonical space
        /// </summary>
        public float[] Xs { get; set; }
    }

    /// <summary>
    /// Motion information extracted from facial keypoints and pose analysis.
    /// Contains all motion parameters needed for facial animation and expression transfer.
    /// </summary>
    internal class MotionInfo
    {
        /// <summary>
        /// Processed pitch rotation angles in degrees
        /// </summary>
        public float[] Pitch { get; set; }
        
        /// <summary>
        /// Processed yaw rotation angles in degrees
        /// </summary>
        public float[] Yaw { get; set; }
        
        /// <summary>
        /// Processed roll rotation angles in degrees
        /// </summary>
        public float[] Roll { get; set; }
        
        /// <summary>
        /// Translation parameters for face positioning (t)
        /// </summary>
        public float[] Translation { get; set; }
        
        /// <summary>
        /// Expression deformation parameters (exp)
        /// </summary>
        public float[] Expression { get; set; }
        
        /// <summary>
        /// Scaling factors for face size normalization
        /// </summary>
        public float[] Scale { get; set; }
        
        /// <summary>
        /// 3D keypoints in canonical face coordinate system (kp)
        /// </summary>
        public float[] Keypoints { get; set; }
        
        /// <summary>
        /// Rotation matrix (R_d) computed from pitch, yaw, and roll angles
        /// </summary>
        public float[,] RotationMatrix { get; set; }
    }

    #endregion
}
