using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine;

namespace LiveTalk.Utils
{
    using Core;
    using API;

    /// <summary>
    /// Comprehensive ONNX model loading and configuration utilities for LiveTalk inference pipeline.
    /// Provides advanced model management including CoreML acceleration, logging integration, asynchronous loading,
    /// cache management, and cross-platform compatibility with optimized session configurations.
    /// All methods handle Unity integration, error recovery, and performance optimization for real-time inference.
    /// </summary>
    internal static class ModelUtils
    {
        #region Private Structures and Fields
        
        // Static configuration fields for ONNX Runtime management
        private static bool _loggingInitialized = false;
        private static readonly Queue<Tuple<Task, string>> _taskQueue = new();
        private static bool _disposeLoadThread = false;
        private static string _cacheDirectory = "";
        private static OrtLoggingLevel _ortLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets the current ONNX Runtime logging level for debugging and diagnostics.
        /// </summary>
        public static OrtLoggingLevel OrtLogLevel { get => _ortLogLevel; }

        #endregion

        #region Public Methods - Initialization and Configuration

        /// <summary>
        /// Initializes the ModelUtils system with ONNX Runtime logging integration and task queue management.
        /// This method sets up Unity logging callbacks, initializes the task queue for asynchronous model loading,
        /// and configures the global logging level for all ONNX operations.
        /// </summary>
        /// <param name="logLevel">The logging level for ONNX Runtime operations (default: WARNING)</param>
        /// <exception cref="InvalidOperationException">Thrown when ONNX Runtime initialization fails</exception>
        public static void Initialize(LogLevel logLevel = LogLevel.WARNING)
        {
            var ortLogLevel = logLevel switch
            {
                LogLevel.VERBOSE => OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                LogLevel.INFO => OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO,
                LogLevel.WARNING => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                LogLevel.ERROR => OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
                _ => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
            };
            _ortLogLevel = ortLogLevel;

            InitializeOnnxLogging();
            _cacheDirectory = GetCoreMLCacheDirectory();
            FileUtils.EnsureDirectoryExists(_cacheDirectory);
            
            // Start background task queue processor for asynchronous model loading
            Task.Run(async() => {
                while (true)
                {
                    bool startedTask = false;
                    if (_taskQueue.Count > 0 && !startedTask)
                    {
                        var task = _taskQueue.Dequeue();
                        SetLoggingParam(task.Item2);
                        startedTask = true;
                        task.Item1.Start();                        
                        await task.Item1;
                        startedTask = false;
                    }
                    else
                    {
                        await Task.Delay(30);
                    }
                    if (_disposeLoadThread && !startedTask)
                    {
                        break;
                    }
                }
            });
        }

        /// <summary>
        /// Sets the logging parameter context for ONNX Runtime operations.
        /// This method marshals the model name to unmanaged memory for use in native logging callbacks.
        /// </summary>
        /// <param name="modelName">The name of the model currently being processed</param>
        /// <exception cref="ArgumentNullException">Thrown when modelName is null</exception>
        /// <exception cref="OutOfMemoryException">Thrown when memory allocation fails</exception>
        public static void SetLoggingParam(string modelName)
        {
            if (modelName == null)
                throw new ArgumentNullException(nameof(modelName));

            // ONNX Runtime allows one environment per process and the library
            // that creates it owns the logging sink. The TTS package
            // initializes first (see LiveTalkAPI.Initialize), so attribution
            // has to go through the buffer its sink actually reads; writing a
            // local one would label nothing.
            QwenTTS.QwenTts.SetOnnxLogContext(modelName);
        }

        /// <summary>
        /// Releases all resources and stops the background task processing thread.
        /// This method should be called during application shutdown to ensure proper cleanup.
        /// </summary>
        public static void Dispose()
        {
            _disposeLoadThread = true;
        }

        #endregion

        #region Public Methods - Model Loading

        /// <summary>
        /// Loads an ONNX model with optimized configuration and hardware acceleration support.
        /// This method handles CoreML acceleration on supported platforms, automatically selects appropriate
        /// execution providers, and provides comprehensive error handling with fallback mechanisms.
        /// </summary>
        /// <param name="config">The LiveTalk configuration containing model paths and settings</param>
        /// <param name="modelConfig">The specific model configuration including precision and execution provider preferences</param>
        /// <returns>A configured InferenceSession ready for inference operations</returns>
        /// <exception cref="ArgumentNullException">Thrown when config or modelConfig is null</exception>
        /// <exception cref="FileNotFoundException">Thrown when the model file cannot be found</exception>
        /// <exception cref="InvalidOperationException">Thrown when model loading fails</exception>
        public static InferenceSession LoadModel(LiveTalkConfig config, ModelConfig modelConfig)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));
            if (modelConfig == null)
                throw new ArgumentNullException(nameof(modelConfig));
            
            if (modelConfig.precision == Precision.INT8)
            {
                // Use CPU if INT8 is enabled for better compatibility
                modelConfig.preferredExecutionProvider = ExecutionProvider.CPU;
            }
            
            string modelPath = GetModelPath(config, modelConfig);
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"{modelConfig.modelName} model not found: {modelPath}");
            
            var sessionOptions = CreateSessionOptions();
            if (modelConfig.preferredExecutionProvider == ExecutionProvider.CoreML) 
            {
                return LoadModelWithCoreML(modelPath, sessionOptions);
            }
            
            // Default CPU execution with optimized settings
            var cpuModel = new InferenceSession(modelPath, sessionOptions);
            Logger.Log($"[ModelUtils] Loaded model with CPU provider: {modelPath}");
            return cpuModel;
        }

        #endregion

        #region Public Methods - Task Management

        /// <summary>
        /// Enqueues a task for asynchronous execution with model context for logging.
        /// This method allows for sequential processing of model loading tasks with proper logging context.
        /// </summary>
        /// <param name="task">The task to enqueue for execution</param>
        /// <param name="modelName">The name of the model associated with this task for logging context</param>
        /// <exception cref="ArgumentNullException">Thrown when task or modelName is null</exception>
        public static void EnqueueTask(Task task, string modelName)
        {
            if (task == null)
                throw new ArgumentNullException(nameof(task));
            if (modelName == null)
                throw new ArgumentNullException(nameof(modelName));
            
            _taskQueue.Enqueue(Tuple.Create(task, modelName));
        }

        #endregion

        #region Private Methods - Logging Configuration

        /// <summary>
        /// Initializes ONNX Runtime. Default OrtEnv only — a custom logger
        /// delegate becomes a dangling native fn ptr after an editor domain reload
        /// (SIGSEGV in UserLoggingSink::SendImpl on the next Session.Run).
        /// </summary>
        private static void InitializeOnnxLogging()
        {
            if (_loggingInitialized) return;
            
            // if ios dont initialize logging
            if (Application.platform == RuntimePlatform.IPhonePlayer)
            {
                _loggingInitialized = true;
                return;
            }

            if (OrtEnv.IsCreated)
            {
                Logger.Log("[ModelUtils] ONNX Runtime environment already created");
                _loggingInitialized = true;
                return;
            }

            try
            {
                _ = OrtEnv.Instance();
                _loggingInitialized = true;
                Logger.Log($"[ModelUtils] ONNX Runtime environment ready (LogLevel: {_ortLogLevel})");
            }
            catch (Exception e)
            {
                Logger.LogError($"[ModelUtils] Failed to initialize ONNX Runtime: {e.Message}");
                _loggingInitialized = true; // Prevent retry loops
            }
        }


        /// <summary>
        /// Logs comprehensive ONNX Runtime version information using reflection for deep introspection.
        /// This method attempts to access internal ONNX Runtime version information through reflection
        /// and provides fallback mechanisms for version detection when direct access is not available.
        /// </summary>
        private static void LogOnnxRuntimeVersion()
        {
            try
            {
                // First, log the assembly version information
                var assembly = typeof(InferenceSession).Assembly;
                var assemblyName = assembly.GetName();
                Logger.Log($"[ModelUtils] ONNX Runtime Assembly Version: {assemblyName.Version}");
                Logger.Log($"[ModelUtils] ONNX Runtime Assembly Location: {assembly.Location}");
                
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
                            // Invoke the delegate to get native library version
                            var method = getVersionStringDelegate.GetType().GetMethod("Invoke");
                            if (method != null)
                            {
                                IntPtr versionPtr = (IntPtr)method.Invoke(getVersionStringDelegate, null);
                                
                                if (versionPtr != IntPtr.Zero)
                                {
                                    string version = Marshal.PtrToStringAnsi(versionPtr);
                                    Logger.Log($"[ModelUtils] ONNX Runtime Native Library Version: {version}");
                                }
                                else
                                {
                                    Logger.LogWarning("[ModelUtils] Failed to get ONNX Runtime version - null pointer returned");
                                }
                            }
                            else
                            {
                                Logger.LogWarning("[ModelUtils] Could not find Invoke method on version string delegate");
                            }
                        }
                        else
                        {
                            Logger.LogWarning("[ModelUtils] OrtGetVersionString delegate is null");
                        }
                    }
                    else
                    {
                        Logger.LogWarning("[ModelUtils] Could not find OrtGetVersionString field in NativeMethods");
                    }
                }
                else
                {
                    Logger.LogWarning("[ModelUtils] Could not find NativeMethods type via reflection");
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[ModelUtils] Error getting ONNX Runtime version via reflection: {e.Message}");
                
                // Fallback: try to get version info from assembly
                try
                {
                    var assembly = typeof(InferenceSession).Assembly;
                    var assemblyName = assembly.GetName();
                    Logger.Log($"[ModelUtils] ONNX Runtime Assembly Version (fallback): {assemblyName.Version}");
                }
                catch (Exception fallbackEx)
                {
                    Logger.LogError($"[ModelUtils] Fallback version detection also failed: {fallbackEx.Message}");
                }
            }
        }

        #endregion

        #region Private Methods - Session Configuration

        /// <summary>
        /// Creates optimized SessionOptions for ONNX Runtime with performance tuning and parallelization.
        /// This method configures advanced settings including graph optimization, execution modes,
        /// memory management, and threading parameters for maximum inference performance.
        /// </summary>
        /// <returns>A fully configured SessionOptions object optimized for LiveTalk inference workloads</returns>
        private static SessionOptions CreateSessionOptions()
        {
            var options = new SessionOptions
            {
                LogSeverityLevel = _ortLogLevel
            };

            if (LiveTalkAPI.Instance.Config.MemoryUsage == MemoryUsage.Optimal)
            {
                options.EnableMemoryPattern = false;
                options.EnableCpuMemArena = false;
                options.IntraOpNumThreads = 1;
            }
            
            return options;
        }

        #endregion

        #region Private Methods - CoreML Support

        /// <summary>
        /// Loads an ONNX model with CoreML acceleration and comprehensive error handling.
        /// This method configures CoreML provider with caching support, handles cache corruption recovery,
        /// and provides fallback mechanisms for maximum compatibility across different Apple devices.
        /// </summary>
        /// <param name="config">The LiveTalk configuration containing cache directory settings</param>
        /// <param name="modelPath">The file path to the ONNX model</param>
        /// <param name="sessionOptions">The base session options to configure with CoreML provider</param>
        /// <returns>An InferenceSession with CoreML acceleration, or null if CoreML setup fails</returns>
        /// <exception cref="InvalidOperationException">Thrown when all CoreML configuration attempts fail</exception>
        private static InferenceSession LoadModelWithCoreML(string modelPath, SessionOptions sessionOptions)
        {
            try
            {
                var coremlOptions = new Dictionary<string, string>
                {
                    ["ModelFormat"] = "MLProgram",
                    ["MLComputeUnits"] = "CPUAndGPU",
                    ["RequireStaticInputShapes"] = "0",
                    ["EnableOnSubgraphs"] = "1",
                };
                
                if (!string.IsNullOrEmpty(_cacheDirectory))
                {
                    coremlOptions["ModelCacheDirectory"] = _cacheDirectory;
                }
                
                sessionOptions.AppendExecutionProvider("CoreML", coremlOptions);
                Logger.Log($"[ModelUtils] CoreML provider configured with caching (cache: {_cacheDirectory})");
                
                // Try creating the session - if it fails due to cache corruption, retry
                try
                {
                    var model = new InferenceSession(modelPath, sessionOptions);
                    Logger.Log($"[ModelUtils] Successfully loaded model with CoreML provider: {modelPath}");
                    return model;
                }
                catch (Exception sessionException)
                {
                    if (sessionException.Message.Contains("Manifest.json") || 
                        sessionException.Message.Contains("coreml_cache") ||
                        sessionException.Message.Contains("manifest does not exist"))
                    {
                        Logger.LogWarning($"[ModelUtils] CoreML cache corruption detected. Retrying: {sessionException.Message}");
                        
                        // Wait for cache cleanup and retry
                        System.Threading.Thread.Sleep(1000);
                        
                        var model = new InferenceSession(modelPath, sessionOptions);
                        Logger.Log($"[ModelUtils] Successfully loaded model with CoreML provider after retrying: {modelPath}");
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
                Logger.LogWarning($"[ModelUtils] CoreML provider configuration failed: {e.Message}");
                
                // Fallback to old CoreML flags approach for compatibility
                try
                {
                    var fallbackOptions = CreateSessionOptions();
                    fallbackOptions.AppendExecutionProvider_CoreML(
                        CoreMLFlags.COREML_FLAG_USE_CPU_AND_GPU | 
                        CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM |
                        CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                    
                    var model = new InferenceSession(modelPath, fallbackOptions);
                    Logger.Log("[ModelUtils] Using fallback CoreML provider (no caching)");
                    return model;
                }
                catch (Exception fallbackException)
                {
                    Logger.LogWarning($"[ModelUtils] CoreML fallback also failed: {fallbackException.Message}. Using CPU provider.");
                }
            }
            return null;
        }

        #endregion

        #region Private Methods - Path and Directory Management

        /// <summary>
        /// Gets the cache directory for CoreML compiled models with automatic path resolution.
        /// This method determines the best location for CoreML model caching based on configuration
        /// and platform-specific storage locations for optimal performance and persistence.
        /// </summary>
        /// <returns>The full path to the CoreML cache directory</returns>
        private static string GetCoreMLCacheDirectory()
        {
            var dataPath = Application.dataPath;
            if (Application.platform == RuntimePlatform.IPhonePlayer)
            {
                dataPath = Application.persistentDataPath; // Use persistent data path for iOS
            }
            return Path.Combine(dataPath, "Models", "coreml_cache");
        }

        /// <summary>
        /// Resolves the full file path for an ONNX model with precision suffix handling.
        /// This method constructs the complete model path including precision-specific suffixes
        /// and validates file existence before returning the path.
        /// </summary>
        /// <param name="config">The LiveTalk configuration containing base model paths</param>
        /// <param name="modelConfig">The model configuration specifying name, path, and precision</param>
        /// <returns>The full path to the ONNX model file</returns>
        /// <exception cref="FileNotFoundException">Thrown when the model file does not exist</exception>
        private static string GetModelPath(LiveTalkConfig config, ModelConfig modelConfig)
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

        #endregion
    }
}
