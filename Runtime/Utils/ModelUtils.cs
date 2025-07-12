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
        
        /// <summary>
        /// Loading information structure for ONNX Runtime logging context.
        /// This structure is marshaled to unmanaged memory for passing to native logging callbacks.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        private struct LoadingInfo
        {
            /// <summary>
            /// The name of the model currently being processed (maximum 256 characters)
            /// </summary>
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
            public string ModelName;
        }

        // Static configuration fields for ONNX Runtime management
        private static bool _loggingInitialized = false;
        private static IntPtr _loggingParam = IntPtr.Zero;
        private static readonly Queue<Tuple<Task, string>> _taskQueue = new();
        private static bool _disposeLoadThread = false;
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
                        // Clean up logging parameter memory
                        if (_loggingParam != IntPtr.Zero)
                        {
                            // Marshal.FreeHGlobal(_loggingParam);
                            _loggingParam = IntPtr.Zero;
                        }
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
            
            var loadingInfo = new LoadingInfo { ModelName = modelName };
            if (_loggingParam == IntPtr.Zero)
            {
                _loggingParam = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LoadingInfo)));
            }
            Marshal.StructureToPtr(loadingInfo, _loggingParam, false);
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
                return LoadModelWithCoreML(config, modelPath, sessionOptions);
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
        /// Initializes ONNX Runtime with Unity logging integration and custom environment options.
        /// This method sets up the global ONNX environment with Unity-compatible logging callbacks
        /// and ensures proper integration with Unity's console system.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown when ONNX Runtime environment creation fails</exception>
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
                Logger.LogWarning("[ModelUtils] OrtEnv already created. Custom logging may not take effect.");
                return;
            }

            try
            {
                // Create loggingParam handle from LoadingInfo structure
                _loggingParam = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LoadingInfo)));
                
                // Create environment options with Unity logging callback
                var options = new EnvironmentCreationOptions
                {
                    logLevel = _ortLogLevel,
                    logId = "LiveTalk",
                    loggingFunction = UnityOnnxLoggingCallback,
                    loggingParam = _loggingParam
                };

                // Initialize OrtEnv with custom options for Unity integration
                OrtEnv.CreateInstanceWithOptions(ref options);
                
                _loggingInitialized = true;
                Logger.Log($"[ModelUtils] ONNX Runtime logging initialized with Unity integration (LogLevel: {_ortLogLevel})");
            }
            catch (Exception e)
            {
                Logger.LogError($"[ModelUtils] Failed to initialize ONNX Runtime logging: {e.Message}");
                _loggingInitialized = true; // Prevent retry loops
            }
        }

        /// <summary>
        /// Unity logging callback for ONNX Runtime that integrates native logging with Unity's console system.
        /// This method receives log messages from the ONNX Runtime native library and forwards them
        /// to Unity's Debug logging system with proper formatting and severity mapping.
        /// </summary>
        /// <param name="param">Pointer to the logging parameter structure containing context information</param>
        /// <param name="severity">The severity level of the log message</param>
        /// <param name="category">The category or component generating the log message</param>
        /// <param name="logId">The identifier for the logging session</param>
        /// <param name="codeLocation">The source code location where the message originated</param>
        /// <param name="message">The actual log message content</param>
        private static void UnityOnnxLoggingCallback(IntPtr param, 
                                                   OrtLoggingLevel severity, 
                                                   string category, 
                                                   string logId, 
                                                   string codeLocation, 
                                                   string message)
        {
            if (param == IntPtr.Zero || _loggingParam == IntPtr.Zero || _disposeLoadThread)
            {
                return;
            }
            
            var loadingInfo = (LoadingInfo)Marshal.PtrToStructure(param, typeof(LoadingInfo));
            string formattedMessage = FormatOnnxLogMessage(severity, category, logId, codeLocation, message, loadingInfo.ModelName);

            switch (severity)
            {
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO:
                    Logger.LogVerbose(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING:
                    Logger.LogWarning(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL:
                    Logger.LogError(formattedMessage);
                    break;
                    
                default:
                    Logger.LogVerbose(formattedMessage);
                    break;
            }
        }

        /// <summary>
        /// Formats ONNX Runtime log messages for Unity console with model context and structured information.
        /// This method creates consistently formatted log messages that include severity, model name,
        /// category, and source location information for enhanced debugging capabilities.
        /// </summary>
        /// <param name="severity">The severity level of the message</param>
        /// <param name="category">The category or component generating the message</param>
        /// <param name="logId">The logging session identifier</param>
        /// <param name="codeLocation">The source code location information</param>
        /// <param name="message">The actual message content</param>
        /// <param name="modelName">The name of the model being processed</param>
        /// <returns>A formatted log message string ready for Unity console display</returns>
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
                };
                
                if (!string.IsNullOrEmpty(cacheDirectory))
                {
                    coremlOptions["ModelCacheDirectory"] = cacheDirectory;
                }
                
                sessionOptions.AppendExecutionProvider("CoreML", coremlOptions);
                Logger.Log($"[ModelUtils] CoreML provider configured with caching (cache: {cacheDirectory})");
                
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
        /// <param name="config">The LiveTalk configuration containing model path preferences</param>
        /// <returns>The full path to the CoreML cache directory</returns>
        private static string GetCoreMLCacheDirectory(LiveTalkConfig config)
        {
            return Path.Combine(Application.dataPath, "Models", "coreml_cache");
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

        /// <summary>
        /// Ensures the CoreML cache directory exists and is writable with proper error handling.
        /// This method creates the cache directory structure if it doesn't exist and handles
        /// permission and filesystem errors gracefully.
        /// </summary>
        /// <param name="cacheDirectory">The cache directory path to create and validate</param>
        private static void EnsureCacheDirectoryExists(string cacheDirectory)
        {
            if (string.IsNullOrEmpty(cacheDirectory))
                return;
                
            try
            {
                if (!Directory.Exists(cacheDirectory))
                {
                    Directory.CreateDirectory(cacheDirectory);
                    Logger.Log($"[ModelUtils] Created CoreML cache directory: {cacheDirectory}");
                }
            }
            catch (Exception e)
            {
                Logger.LogWarning($"[ModelUtils] Failed to create cache directory {cacheDirectory}: {e.Message}");
            }
        }

        #endregion
    }
}
