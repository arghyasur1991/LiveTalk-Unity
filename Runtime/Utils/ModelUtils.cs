using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

namespace LiveTalk.Utils
{
    using Core;
    using API;
    /// <summary>
    /// Utility functions for ONNX model loading and configuration in MuseTalk
    /// </summary>
    internal static class ModelUtils
    {
        /// <summary>
        /// Create optimized session options for ONNX Runtime
        /// OPTIMIZED: Enable all performance optimizations
        /// </summary>
        private static SessionOptions CreateSessionOptions(LiveTalkConfig config)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_PARALLEL,

                EnableMemoryPattern = true,
                EnableCpuMemArena = true,
                InterOpNumThreads = Environment.ProcessorCount,
                IntraOpNumThreads = Environment.ProcessorCount
            };
            
            options.AddSessionConfigEntry("session.disable_prepacking", "0"); // Enable weight prepacking
            options.AddSessionConfigEntry("session.use_env_allocators", "1"); // Use environment allocators
            
            options.AddSessionConfigEntry("session.intra_op_param", ""); // Let ORT auto-tune
            options.AddSessionConfigEntry("session.inter_op_param", ""); // Let ORT auto-tune
            
            try
            {
                // Try CUDA first (NVIDIA GPUs) with optimized settings
                var cudaOptions = new Dictionary<string, string>
                {
                    ["device_id"] = "0",
                    ["arena_extend_strategy"] = "kSameAsRequested", // Optimize memory allocation
                    ["gpu_mem_limit"] = "0", // Use all available GPU memory
                    ["cudnn_conv_algo_search"] = "EXHAUSTIVE", // Find best conv algorithms
                    ["do_copy_in_default_stream"] = "1", // Optimize memory transfers
                };
                
                options.AppendExecutionProvider("CUDAExecutionProvider", cudaOptions);
                Debug.Log("[MuseTalkInference] CUDA GPU provider enabled with optimizations");
            }
            catch (Exception)
            {
                try
                {
                    // Try DirectML (Windows GPU acceleration)
                    options.AppendExecutionProvider_DML(0);
                    Debug.Log("[MuseTalkInference] DirectML GPU provider enabled");
                }
                catch (Exception)
                {
                    // Debug.Log("[MuseTalkInference] Using CPU execution provider (GPU not available)");
                }
            }
            
            return options;
        }

        /// <summary>
        /// Configure CoreML provider with caching and optimization options
        /// </summary>
        private static void ConfigureCoreMLProvider(SessionOptions sessionOptions, LiveTalkConfig config, ModelConfig modelConfig)
        {
            try
            {
                // Set up CoreML cache directory for faster model loading
                string cacheDirectory = GetCoreMLCacheDirectory(config);
                
                // Configure CoreML options using the newer API approach
                var coremlOptions = new Dictionary<string, string>
                {
                    ["ModelFormat"] = "MLProgram", // Use MLProgram format for better performance (iOS 15+/macOS 12+)
                    ["MLComputeUnits"] = "ALL",    // Use all available compute units (CPU, GPU, Neural Engine)
                    ["EnableOnSubgraphs"] = "1",   // Enable on subgraphs
                };
                
                // Add cache directory if available
                if (!string.IsNullOrEmpty(cacheDirectory))
                {
                    coremlOptions["ModelCacheDirectory"] = cacheDirectory;
                    Debug.Log($"[ModelUtils] CoreML cache enabled: {cacheDirectory}");
                }
                
                // Use the newer generic provider API
                sessionOptions.AppendExecutionProvider("CoreMLExecutionProvider", coremlOptions);
                Debug.Log($"[ModelUtils] CoreML provider configured for {modelConfig.modelName} with caching enabled");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[ModelUtils] Failed to configure CoreML with advanced options: {e.Message}");
                try
                {
                    // Fallback to older flag-based method with basic settings
                    // sessionOptions.AppendExecutionProvider_CoreML();
                    Debug.Log($"[ModelUtils] CoreML provider configured using fallback method for {modelConfig.modelName}");
                }
                catch (Exception fallbackException)
                {
                    Debug.LogError($"[ModelUtils] Failed to configure CoreML provider: {fallbackException.Message}");
                    throw;
                }
            }
        }

        /// <summary>
        /// Get or create CoreML cache directory
        /// </summary>
        private static string GetCoreMLCacheDirectory(LiveTalkConfig config)
        {
            try
            {
                // Use config-specific cache directory if available
                string baseCacheDir;
                if (!string.IsNullOrEmpty(config?.ModelPath))
                {
                    baseCacheDir = Path.Combine(config.ModelPath, "coreml_cache");
                }
                else
                {
                    // Use persistent data path as fallback
                    baseCacheDir = Path.Combine(Application.persistentDataPath, "LiveTalk", "coreml_cache");
                }

                // Create directory if it doesn't exist
                if (!Directory.Exists(baseCacheDir))
                {
                    Directory.CreateDirectory(baseCacheDir);
                }

                return baseCacheDir;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[ModelUtils] Failed to create CoreML cache directory: {e.Message}");
                return string.Empty; // Disable caching if directory creation fails
            }
        }

        public static InferenceSession LoadModel(LiveTalkConfig config, ModelConfig modelConfig)
        {
            string modelPath = GetModelPath(config, modelConfig);
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"{modelConfig.modelName} model not found: {modelPath}");
            
            var sessionOptions = CreateSessionOptions(config);
            
            if (modelConfig.preferredExecutionProvider == ExecutionProvider.CoreML && 
                !IsInt8Enabled(config, modelConfig)) // Use CoreML if INT8 is not enabled
            {
                ConfigureCoreMLProvider(sessionOptions, config, modelConfig);
            }
            
            var model = new InferenceSession(modelPath, sessionOptions);
            Debug.Log($"[ModelUtils] Loaded model: {modelPath}");
            return model;
        }

        private static bool IsInt8Enabled(LiveTalkConfig config, ModelConfig modelConfig)
        {
            return config.UseINT8 && modelConfig.precision == Precision.INT8;
        }

        /// <summary>
        /// Get model file path with optimal quality/performance balance
        /// QUALITY OPTIMIZATION: Automatically use FP32 for VAE models to preserve image quality
        /// </summary>
        public static string GetModelPath(LiveTalkConfig config, ModelConfig modelConfig)
        {
            bool isVersionIndependent = modelConfig.version == "";
            string precisionSuffix = modelConfig.precision == Precision.FP32 ? "" : 
                                        $"_{modelConfig.precision.ToString().ToLower()}";

            string modelName = modelConfig.modelName;
            modelName += isVersionIndependent ? "" : $"_{modelConfig.version}";
            modelName += precisionSuffix;
            string modelPath = Path.Combine(config.ModelPath, $"{modelName}.onnx");
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"{modelConfig.modelName} model not found: {modelPath}");
            }
            return modelPath;
        }

        /// <summary>
        /// Load mask template texture from Resources or StreamingAssets
        /// Matches Python: mask_crop = cv2.imread('mask_template.png')
        /// </summary>
        public static Frame LoadMaskTemplate(LiveTalkConfig config)
        {
            try
            {
                // First try to load from Resources
                var maskTexture = Resources.Load<Texture2D>("mask_template");
                if (maskTexture != null)
                {
                    Debug.Log("[ModelUtils] Loaded mask template from Resources");
                    return TextureUtils.Texture2DToFrame(TextureUtils.ConvertTexture2DToRGB24(maskTexture));
                }
                
                // Try to load from StreamingAssets
                string maskPath = System.IO.Path.Combine(Application.streamingAssetsPath, "mask_template.png");
                if (System.IO.File.Exists(maskPath))
                {
                    byte[] fileData = System.IO.File.ReadAllBytes(maskPath);
                    var texture = new Texture2D(2, 2);
                    if (texture.LoadImage(fileData))
                    {                        
                        Debug.Log("[ModelUtils] Loaded mask template from StreamingAssets");
                        return TextureUtils.Texture2DToFrame(TextureUtils.ConvertTexture2DToRGB24(texture));
                    }
                    else
                    {
                        UnityEngine.Object.DestroyImmediate(texture);
                    }
                }
                
                // Try to load from config model path
                if (!string.IsNullOrEmpty(config?.ModelPath))
                {
                    string configMaskPath = System.IO.Path.Combine(config.ModelPath, "mask_template.png");
                    if (System.IO.File.Exists(configMaskPath))
                    {
                        byte[] fileData = System.IO.File.ReadAllBytes(configMaskPath);
                        var texture = new Texture2D(2, 2);
                        if (texture.LoadImage(fileData))
                        {
                            Debug.Log($"[ModelUtils] Loaded mask template from config path: {configMaskPath}");
                            return TextureUtils.Texture2DToFrame(TextureUtils.ConvertTexture2DToRGB24(texture));
                        }
                        else
                        {
                            UnityEngine.Object.DestroyImmediate(texture);
                        }
                    }
                }
                
                Debug.LogWarning("[ModelUtils] Could not find mask_template.png in Resources, StreamingAssets, or config path. Will use default mask.");
                return new Frame(null, 0, 0);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[ModelUtils] Error loading mask template: {e.Message}");
                return new Frame(null, 0, 0);
            }
        }
    }
} 