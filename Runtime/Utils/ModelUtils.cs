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
    using System.Threading.Tasks;

    /// <summary>
    /// Utility functions for ONNX model loading and configuration in MuseTalk
    /// </summary>
    internal static class ModelUtils
    {
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        private struct LoadingInfo
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
            public string ModelName;
        }

        private static bool _loggingInitialized = false;
        private static IntPtr _loggingParam = IntPtr.Zero;

        private static readonly Queue<Tuple<Task, string>> _taskQueue = new();
        private static bool _disposeLoadThread = false;

        /// <summary>
        /// Initialize ONNX Runtime with Unity logging integration
        /// Automatically called by LoadModel but can be called explicitly
        /// </summary>
        public static void InitializeOnnxLogging(OrtLoggingLevel logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
        {
            if (_loggingInitialized) return;

            if (OrtEnv.IsCreated)
            {
                Debug.LogWarning("[ModelUtils] OrtEnv already created. Custom logging may not take effect.");
                return;
            }

            try
            {
                // Create loggingParam handle from _loadingInfo
                _loggingParam = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LoadingInfo)));
                // Create environment options with Unity logging callback
                var options = new EnvironmentCreationOptions
                {
                    logLevel = logLevel,
                    logId = "LiveTalk",
                    loggingFunction = UnityOnnxLoggingCallback,
                    loggingParam = _loggingParam
                };

                // Initialize OrtEnv with custom options
                OrtEnv.CreateInstanceWithOptions(ref options);
                
                _loggingInitialized = true;
                Debug.Log($"[ModelUtils] ONNX Runtime logging initialized with Unity integration (LogLevel: {logLevel})");
            }
            catch (Exception e)
            {
                Debug.LogError($"[ModelUtils] Failed to initialize ONNX Runtime logging: {e.Message}");
                _loggingInitialized = true; // Prevent retry loops
            }
        }

        public static void Initialize()
        {
            InitializeOnnxLogging(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
            Task.Run(async() => {
                while (true)
                {
                    if (_taskQueue.Count > 0)
                    {
                        var task = _taskQueue.Dequeue();
                        var loadingInfo = new LoadingInfo { ModelName = task.Item2 };
                        Marshal.StructureToPtr(loadingInfo, _loggingParam, false);
                        
                        await task.Item1;
                    }
                    if (_disposeLoadThread)
                    {
                        break;
                    }
                }
            });
        }

        public static void EnqueueTask(Task task, string modelName)
        {
            _taskQueue.Enqueue(Tuple.Create(task, modelName));
        }

        public static void Dispose()
        {
            _disposeLoadThread = true;
            Marshal.FreeHGlobal(_loggingParam);
        }

        /// <summary>
        /// Unity logging callback for ONNX Runtime
        /// </summary>
        private static void UnityOnnxLoggingCallback(IntPtr param, 
                                                   OrtLoggingLevel severity, 
                                                   string category, 
                                                   string logId, 
                                                   string codeLocation, 
                                                   string message)
        {
            var loadingInfo = (LoadingInfo)Marshal.PtrToStructure(param, typeof(LoadingInfo));
            // Debug.Log($"[ModelUtils] ONNX Runtime logging callback: {severity} {category} {logId} {codeLocation} {message}");
            string formattedMessage = FormatOnnxLogMessage(severity, category, logId, codeLocation, message, loadingInfo.ModelName);

            switch (severity)
            {
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO:
                    Debug.Log(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING:
                    Debug.LogWarning(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL:
                    Debug.LogError(formattedMessage);
                    break;
                    
                default:
                    Debug.Log(formattedMessage);
                    break;
            }
        }

        /// <summary>
        /// Format ONNX Runtime log message for Unity console
        /// </summary>
        private static string FormatOnnxLogMessage(OrtLoggingLevel severity, 
                                                 string category, 
                                                 string logId, 
                                                 string codeLocation, 
                                                 string message,
                                                 string modelName)
        {
            string severityStr = severity switch
            {
                OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE => "VERBOSE",
                OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO => "INFO", 
                OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING => "WARN",
                OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR => "ERROR",
                OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL => "FATAL",
                _ => "UNKNOWN"
            };

            string cleanCategory = !string.IsNullOrEmpty(category) ? $"[{category}]" : "";
            string cleanCodeLocation = !string.IsNullOrEmpty(codeLocation) ? $" ({codeLocation})" : "";
            
            return $"[ONNX-{severityStr}][{modelName}]{cleanCategory} {message}{cleanCodeLocation}";
        }

        /// <summary>
        /// Enable verbose ONNX Runtime logging for debugging
        /// Must be called before any ONNX operations
        /// </summary>
        public static void EnableVerboseLogging()
        {
            if (_loggingInitialized)
            {
                Debug.LogWarning("[ModelUtils] Logging already initialized. Verbose logging setting may not take effect.");
                return;
            }
            InitializeOnnxLogging(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
        }
        /// <summary>
        /// Log model metadata including cache keys for debugging
        /// </summary>
        public static void LogModelMetadata(string modelPath)
        {
            try
            {
                using (var session = new InferenceSession(modelPath))
                {
                    var metadata = session.ModelMetadata;
                    Debug.Log($"[ModelUtils] Model Metadata for {Path.GetFileName(modelPath)}:");
                    Debug.Log($"  Producer: {metadata.ProducerName}");
                    Debug.Log($"  Version: {metadata.Version}");
                    Debug.Log($"  Domain: {metadata.Domain}");
                    
                    var customMetadata = metadata.CustomMetadataMap;
                    if (customMetadata != null && customMetadata.Count > 0)
                    {
                        Debug.Log("  Custom Metadata:");
                        foreach (var kvp in customMetadata)
                        {
                            Debug.Log($"    {kvp.Key}: {kvp.Value}");
                            if (kvp.Key == "CACHE_KEY")
                            {
                                Debug.Log($"    ✅ Found CACHE_KEY: {kvp.Value}");
                            }
                        }
                    }
                    else
                    {
                        Debug.Log("  ⚠️ No custom metadata found - cache will use file path hash");
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[ModelUtils] Error reading model metadata from {modelPath}: {e.Message}");
            }
        }

        /// <summary>
        /// Log the ONNX Runtime version information
        /// </summary>
        public static void LogOnnxRuntimeVersion()
        {
            try
            {
                // First, log the assembly version
                var assembly = typeof(InferenceSession).Assembly;
                var assemblyName = assembly.GetName();
                Debug.Log($"[ModelUtils] ONNX Runtime Assembly Version: {assemblyName.Version}");
                Debug.Log($"[ModelUtils] ONNX Runtime Assembly Location: {assembly.Location}");
                
                // Use reflection to access the internal NativeMethods.OrtGetVersionString
                var nativeMethodsType = typeof(InferenceSession).Assembly.GetType("Microsoft.ML.OnnxRuntime.NativeMethods");
                if (nativeMethodsType != null)
                {
                    var getVersionStringField = nativeMethodsType.GetField("OrtGetVersionString", 
                        System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
                    
                    if (getVersionStringField != null)
                    {
                        var getVersionStringDelegate = getVersionStringField.GetValue(null);
                        if (getVersionStringDelegate != null)
                        {
                            // Invoke the delegate
                            var method = getVersionStringDelegate.GetType().GetMethod("Invoke");
                            if (method != null)
                            {
                                IntPtr versionPtr = (IntPtr)method.Invoke(getVersionStringDelegate, null);
                                
                                if (versionPtr != IntPtr.Zero)
                                {
                                    string version = Marshal.PtrToStringAnsi(versionPtr);
                                    Debug.Log($"[ModelUtils] ONNX Runtime Native Library Version: {version}");
                                }
                                else
                                {
                                    Debug.LogWarning("[ModelUtils] Failed to get ONNX Runtime version - null pointer returned");
                                }
                            }
                            else
                            {
                                Debug.LogWarning("[ModelUtils] Could not find Invoke method on version string delegate");
                            }
                        }
                        else
                        {
                            Debug.LogWarning("[ModelUtils] OrtGetVersionString delegate is null");
                        }
                    }
                    else
                    {
                        Debug.LogWarning("[ModelUtils] Could not find OrtGetVersionString field in NativeMethods");
                    }
                }
                else
                {
                    Debug.LogWarning("[ModelUtils] Could not find NativeMethods type via reflection");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[ModelUtils] Error getting ONNX Runtime version via reflection: {e.Message}");
                
                // Fallback: try to get version info from assembly
                try
                {
                    var assembly = typeof(InferenceSession).Assembly;
                    var assemblyName = assembly.GetName();
                    Debug.Log($"[ModelUtils] ONNX Runtime Assembly Version (fallback): {assemblyName.Version}");
                }
                catch (Exception fallbackEx)
                {
                    Debug.LogError($"[ModelUtils] Fallback version detection also failed: {fallbackEx.Message}");
                }
            }
        }

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
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
            
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

        private static InferenceSession LoadModelWithCoreML(LiveTalkConfig config, string modelPath, SessionOptions sessionOptions)
        {
            try
            {
                // Configure CoreML provider with caching support using dictionary API
                string cacheDirectory = GetCoreMLCacheDirectory(config);
                
                // Ensure cache directory exists and is writable
                EnsureCacheDirectoryExists(cacheDirectory);
                
                var coremlOptions = new Dictionary<string, string>
                {
                    ["ModelFormat"] = "MLProgram",
                    ["MLComputeUnits"] = "CPUAndGPU",
                    ["RequireStaticInputShapes"] = "0",
                    ["EnableOnSubgraphs"] = "1",

                    // ["SpecializationStrategy"] = "FastPrediction",
                    // ["AllowLowPrecisionAccumulationOnGPU"] = "1",
                    ["ProfileComputePlan"] = "1"
                };
                // Enable profiling for specific models if verbose logging is enabled
                if (modelPath.Contains("warping"))
                {
                    // sessionOptions.EnableProfiling = true;
                    Debug.Log($"[ModelUtils] Profiling enabled for model: {Path.GetFileName(modelPath)}");
                }
                
                if (!string.IsNullOrEmpty(cacheDirectory))
                {
                    coremlOptions["ModelCacheDirectory"] = cacheDirectory;
                }
                
                sessionOptions.AppendExecutionProvider("CoreML", coremlOptions);
                Debug.Log($"[ModelUtils] CoreML provider configured with caching (cache: {cacheDirectory})");
                
                // Try creating the session - if it fails due to cache corruption, retry
                try
                {
                    var model = new InferenceSession(modelPath, sessionOptions);
                    Debug.Log($"[ModelUtils] Successfully loaded model with CoreML provider: {modelPath}");
                    return model;
                }
                catch (Exception sessionException)
                {
                    if (sessionException.Message.Contains("Manifest.json") || 
                        sessionException.Message.Contains("coreml_cache") ||
                        sessionException.Message.Contains("manifest does not exist"))
                    {
                        Debug.LogWarning($"[ModelUtils] CoreML cache corruption detected. Retrying: {sessionException.Message}");
                        
                        // wait for 1 second
                        System.Threading.Thread.Sleep(1000);
                        
                        var model = new InferenceSession(modelPath, sessionOptions);
                        Debug.Log($"[ModelUtils] Successfully loaded model with CoreML provider after retrying: {modelPath}");
                        return model;
                    }
                    else
                    {
                        throw; // Re-throw if it's not a cache-related issue
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[ModelUtils] CoreML provider configuration failed: {e.Message}");
                
                // Fallback to old CoreML flags approach
                try
                {
                    var fallbackOptions = CreateSessionOptions(config);
                    fallbackOptions.AppendExecutionProvider_CoreML(
                        CoreMLFlags.COREML_FLAG_USE_CPU_AND_GPU | 
                        CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM |
                        CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                    
                    var model = new InferenceSession(modelPath, fallbackOptions);
                    Debug.Log("[ModelUtils] Using fallback CoreML provider (no caching)");
                    return model;
                }
                catch (Exception fallbackException)
                {
                    Debug.LogWarning($"[ModelUtils] CoreML fallback also failed: {fallbackException.Message}. Using CPU provider.");
                }
            }
            return null;
        }

        public static InferenceSession LoadModel(LiveTalkConfig config, ModelConfig modelConfig)
        {
            // Initialize ONNX logging first if not already done
            InitializeOnnxLogging(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
            
            if (modelConfig.precision == Precision.INT8)
            {
                // Use CPU if INT8 is enabled
                modelConfig.preferredExecutionProvider = ExecutionProvider.CPU;
            }
            
            string modelPath = GetModelPath(config, modelConfig);
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"{modelConfig.modelName} model not found: {modelPath}");
            
            var sessionOptions = CreateSessionOptions(config);
            if (modelConfig.preferredExecutionProvider == ExecutionProvider.CoreML) 
            {
                return LoadModelWithCoreML(config, modelPath, sessionOptions);
            }
            
            // Default CPU execution
            var cpuModel = new InferenceSession(modelPath, sessionOptions);
            Debug.Log($"[ModelUtils] Loaded model with CPU provider: {modelPath}");
            return cpuModel;
        }

        /// <summary>
        /// Get the cache directory for CoreML compiled models
        /// </summary>
        private static string GetCoreMLCacheDirectory(LiveTalkConfig config)
        {
            if (!string.IsNullOrEmpty(config?.ModelPath))
            {
                return Path.Combine(config.ModelPath, "coreml_cache");
            }
            return Path.Combine(Application.persistentDataPath, "coreml_cache");
        }

        /// <summary>
        /// Get model file path
        /// </summary>
        public static string GetModelPath(LiveTalkConfig config, ModelConfig modelConfig)
        {
            string precisionSuffix = modelConfig.precision == Precision.FP32 ? "" : 
                                        $"_{modelConfig.precision.ToString().ToLower()}";

            string modelName = modelConfig.modelName;
            modelName += precisionSuffix;
            string modelPath = Path.Combine(config.ModelPath, modelConfig.modelRelativePath, $"{modelName}.onnx");
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

        /// <summary>
        /// Ensure the CoreML cache directory exists and is writable
        /// </summary>
        private static void EnsureCacheDirectoryExists(string cacheDirectory)
        {
            if (string.IsNullOrEmpty(cacheDirectory))
                return;
                
            try
            {
                if (!Directory.Exists(cacheDirectory))
                {
                    Directory.CreateDirectory(cacheDirectory);
                    Debug.Log($"[ModelUtils] Created CoreML cache directory: {cacheDirectory}");
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[ModelUtils] Failed to create cache directory {cacheDirectory}: {e.Message}");
            }
        }

        /// <summary>
        /// Clean corrupted CoreML cache directory
        /// </summary>
        private static void CleanCorruptedCoreMLCache(string cacheDirectory)
        {
            if (string.IsNullOrEmpty(cacheDirectory) || !Directory.Exists(cacheDirectory))
                return;
                
            try
            {
                Debug.Log($"[ModelUtils] Cleaning corrupted CoreML cache: {cacheDirectory}");
                Directory.Delete(cacheDirectory, true);
                
                // Recreate the directory
                Directory.CreateDirectory(cacheDirectory);
                Debug.Log($"[ModelUtils] CoreML cache cleaned and recreated: {cacheDirectory}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[ModelUtils] Failed to clean CoreML cache {cacheDirectory}: {e.Message}");
            }
        }
    }
} 