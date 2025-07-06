using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace LiveTalk.Core
{
    using API;
    using Utils;
    
    /// <summary>
    /// Represents an ONNX model wrapper for LiveTalk inference operations.
    /// Provides asynchronous loading, input/output management, and execution capabilities.
    /// </summary>
    internal class Model : IDisposable
    {
        #region Private Fields
        
        private readonly ModelConfig _config;
        private InferenceSession _session;
        private List<string> _inputNames = new();
        private List<NamedOnnxValue> _inputs;
        private List<NamedOnnxValue> _preallocatedOutputs;
        private readonly Task<InferenceSession> _loadTask;
        private bool _disposed = false;
        
        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the Model class with the specified configuration.
        /// </summary>
        /// <param name="config">The LiveTalk configuration containing model paths and settings</param>
        /// <param name="modelName">The name of the model file to load</param>
        /// <param name="modelRelativePath">The relative path to the model directory</param>
        /// <param name="preferredExecutionProvider">The preferred execution provider (CPU, CUDA, CoreML)</param>
        /// <param name="precision">The precision mode for the model (FP32, FP16, INT8)</param>
        /// <exception cref="ArgumentNullException">Thrown when config is null</exception>
        public Model(
            LiveTalkConfig config, 
            string modelName, 
            string modelRelativePath,
            ExecutionProvider preferredExecutionProvider = ExecutionProvider.CPU, 
            Precision precision = Precision.FP32)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));
                
            _config = new ModelConfig(modelName, modelRelativePath, preferredExecutionProvider, precision);
            _loadTask = LoadModel(config);
        }

        #endregion

        #region Public Methods - Input Loading

        /// <summary>
        /// Asynchronously loads a tensor input at the specified index.
        /// </summary>
        /// <param name="index">The index of the input to load</param>
        /// <param name="inputTensor">The tensor to load as input</param>
        /// <returns>A task that represents the asynchronous input loading operation</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range</exception>
        /// <exception cref="ArgumentNullException">Thrown when inputTensor is null</exception>
        public async Task LoadInput<T>(int index, Tensor<T> inputTensor)
        {
            if (inputTensor == null)
                throw new ArgumentNullException(nameof(inputTensor));
                
            await _loadTask;
            
            if (index < 0 || index >= _inputNames.Count)
                throw new ArgumentOutOfRangeException(nameof(index), $"Index must be between 0 and {_inputNames.Count - 1}");
                
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[index], inputTensor));
        }
        
        /// <summary>
        /// Asynchronously loads multiple float tensor inputs in order.
        /// The number of input tensors must match the model's expected input count.
        /// </summary>
        /// <param name="inputTensors">The list of float tensors to load as inputs</param>
        /// <returns>A task that represents the asynchronous input loading operation</returns>
        /// <exception cref="ArgumentNullException">Thrown when inputTensors is null</exception>
        /// <exception cref="ArgumentException">Thrown when input tensor count doesn't match model requirements</exception>
        public async Task LoadInputs(List<Tensor<float>> inputTensors)
        {
            if (inputTensors == null)
                throw new ArgumentNullException(nameof(inputTensors));
                
            await _loadTask;
            
            var inputNames = _session.InputMetadata.Keys.ToArray();
            if (inputTensors.Count != inputNames.Length)
            {
                throw new ArgumentException(
                    $"Input tensor count mismatch: provided {inputTensors.Count}, expected {inputNames.Length}",
                    nameof(inputTensors));
            }
            
            for (int i = 0; i < inputTensors.Count; i++)
            {
                if (inputTensors[i] == null)
                    throw new ArgumentNullException($"inputTensors[{i}]", "Input tensor cannot be null");
                    
                _inputs.Add(NamedOnnxValue.CreateFromTensor(inputNames[i], inputTensors[i]));
            }
        }

        #endregion

        #region Public Methods - Model Execution

        /// <summary>
        /// Asynchronously runs the model with the provided input tensors and returns the results.
        /// Uses preallocated output buffers for improved performance.
        /// </summary>
        /// <param name="inputTensors">The list of input tensors for model execution</param>
        /// <returns>A task containing the list of named output values from model execution</returns>
        /// <exception cref="ArgumentNullException">Thrown when inputTensors is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when model execution fails</exception>
        public async Task<List<NamedOnnxValue>> Run(List<Tensor<float>> inputTensors)
        {
            if (inputTensors == null)
                throw new ArgumentNullException(nameof(inputTensors));
                
            await _loadTask;
            await LoadInputs(inputTensors);
            return await Run();
        }

        /// <summary>
        /// Asynchronously runs the model with previously loaded inputs.
        /// Uses preallocated output buffers for optimal performance.
        /// </summary>
        /// <returns>A task containing the list of named output values from model execution</returns>
        /// <exception cref="InvalidOperationException">Thrown when no inputs are loaded or model execution fails</exception>
        public async Task<List<NamedOnnxValue>> Run()
        {
            await _loadTask;
            
            if (_inputs == null || _inputs.Count == 0)
                throw new InvalidOperationException("No inputs loaded. Call LoadInputs() or LoadInput() first.");
            
            var start = System.Diagnostics.Stopwatch.StartNew();
            ModelUtils.SetLoggingParam(_config.modelName);
            
            var runOptions = new RunOptions
            {
                LogSeverityLevel = ModelUtils.OrtLogLevel
            };
            
            try
            {
                _session.Run(_inputs, _preallocatedOutputs, runOptions);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Model execution failed for {_config.modelName}: {ex.Message}", ex);
            }
            finally
            {
                var elapsed = start.ElapsedMilliseconds;
                Logger.Log($"[Model] {_config.modelName} execution completed in {elapsed}ms");
            }

            return _preallocatedOutputs;
        }

        /// <summary>
        /// Asynchronously runs the model with the provided input tensors and returns disposable results.
        /// Creates new output tensors for each execution (higher memory usage but safer for concurrent access).
        /// </summary>
        /// <param name="inputTensors">The list of input tensors for model execution</param>
        /// <returns>A task containing the disposable collection of named output values</returns>
        /// <exception cref="ArgumentNullException">Thrown when inputTensors is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when model execution fails</exception>
        public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunDisposable(
            List<Tensor<float>> inputTensors)
        {
            if (inputTensors == null)
                throw new ArgumentNullException(nameof(inputTensors));
                
            await _loadTask;
            await LoadInputs(inputTensors);
            return await RunDisposable();
        }

        /// <summary>
        /// Asynchronously runs the model with previously loaded inputs and returns disposable results.
        /// Creates new output tensors for each execution (higher memory usage but safer for concurrent access).
        /// </summary>
        /// <returns>A task containing the disposable collection of named output values</returns>
        /// <exception cref="InvalidOperationException">Thrown when no inputs are loaded or model execution fails</exception>
        public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunDisposable()
        {
            await _loadTask;
            
            if (_inputs == null || _inputs.Count == 0)
                throw new InvalidOperationException("No inputs loaded. Call LoadInputs() or LoadInput() first.");
            
            var start = System.Diagnostics.Stopwatch.StartNew();
            ModelUtils.SetLoggingParam(_config.modelName);
            
            var runOptions = new RunOptions
            {
                LogSeverityLevel = ModelUtils.OrtLogLevel
            };
            
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            try
            {
                results = _session.Run(_inputs, _session.OutputNames, runOptions);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Model execution failed for {_config.modelName}: {ex.Message}", ex);
            }
            finally
            {
                var elapsed = start.ElapsedMilliseconds;
                Logger.Log($"[Model] {_config.modelName} execution completed in {elapsed}ms");
            }
            
            return results;
        }

        #endregion

        #region Public Methods - Output Management

        /// <summary>
        /// Asynchronously retrieves a preallocated output tensor by name.
        /// </summary>
        /// <typeparam name="T">The tensor element type</typeparam>
        /// <param name="outputName">The name of the output tensor to retrieve</param>
        /// <returns>A task containing the requested preallocated output tensor</returns>
        /// <exception cref="ArgumentNullException">Thrown when outputName is null or empty</exception>
        /// <exception cref="ArgumentException">Thrown when the specified output is not found</exception>
        public async Task<Tensor<T>> GetPreallocatedOutput<T>(string outputName)
        {
            if (string.IsNullOrEmpty(outputName))
                throw new ArgumentNullException(nameof(outputName));
                
            await _loadTask;
            
            var output = _preallocatedOutputs.FirstOrDefault(o => o.Name == outputName);
            if (output != null)
            {
                return output.AsTensor<T>();
            }
            
            throw new ArgumentException($"Output '{outputName}' not found in preallocated outputs", nameof(outputName));
        }

        /// <summary>
        /// Asynchronously retrieves all preallocated output tensors.
        /// </summary>
        /// <returns>A task containing the list of all preallocated output tensors</returns>
        public async Task<List<NamedOnnxValue>> GetPreallocatedOutputs()
        {
            await _loadTask;
            return _preallocatedOutputs;
        }

        /// <summary>
        /// Asynchronously updates the dimensions of a preallocated output tensor.
        /// Useful for handling dynamic shapes in model outputs.
        /// </summary>
        /// <param name="outputName">The name of the output tensor to update</param>
        /// <param name="newDimensions">The new dimensions for the output tensor</param>
        /// <returns>A task that represents the asynchronous dimension update operation</returns>
        /// <exception cref="ArgumentNullException">Thrown when outputName is null or newDimensions is null</exception>
        /// <exception cref="ArgumentException">Thrown when the specified output is not found</exception>
        public async Task UpdateOutputDimensions(string outputName, int[] newDimensions)
        {
            if (string.IsNullOrEmpty(outputName))
                throw new ArgumentNullException(nameof(outputName));
            if (newDimensions == null)
                throw new ArgumentNullException(nameof(newDimensions));
                
            await _loadTask;
            
            var outputIndex = _preallocatedOutputs.FindIndex(o => o.Name == outputName);
            if (outputIndex >= 0)
            {
                // Recreate tensor with new dimensions
                var newTensor = new DenseTensor<float>(newDimensions);
                _preallocatedOutputs[outputIndex] = NamedOnnxValue.CreateFromTensor(outputName, newTensor);
            }
            else
            {
                throw new ArgumentException($"Output '{outputName}' not found in preallocated outputs", nameof(outputName));
            }
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Creates a preallocated tensor for the specified output with given dimensions.
        /// Replaces dynamic dimensions (-1) with 1 for tensor initialization.
        /// </summary>
        /// <typeparam name="T">The tensor element type (float, long, etc.)</typeparam>
        /// <param name="outputName">The name of the output tensor</param>
        /// <param name="dimensions">The tensor dimensions array</param>
        private void CreatePreallocatedTensor<T>(string outputName, int[] dimensions)
        {
            // Replace dynamic dimensions with 1 for tensor creation
            dimensions = dimensions.Select(d => d == -1 ? 1 : d).ToArray();
            var tensor = new DenseTensor<T>(dimensions);
            _preallocatedOutputs.Add(NamedOnnxValue.CreateFromTensor(outputName, tensor));
        }

        /// <summary>
        /// Asynchronously loads the ONNX model and initializes input/output metadata.
        /// Sets up preallocated output buffers for improved performance.
        /// </summary>
        /// <param name="config">The LiveTalk configuration for model loading</param>
        /// <returns>A task that represents the asynchronous model loading operation</returns>
        /// <exception cref="InvalidOperationException">Thrown when model loading fails</exception>
        private async Task<InferenceSession> LoadModel(LiveTalkConfig config)
        {
            var task = new Task(() => {
                _session = ModelUtils.LoadModel(config, _config);
            });
            
            ModelUtils.EnqueueTask(task, _config.modelName);
            await task;
            
            if (_session == null)
                throw new InvalidOperationException($"Failed to load model: {_config.modelName}");
            
            // Initialize input metadata
            _inputNames = _session.InputMetadata.Keys.ToList();
            _inputs = new List<NamedOnnxValue>(_inputNames.Count);
            
            // Pre-allocate output buffers based on output metadata
            _preallocatedOutputs = new List<NamedOnnxValue>();
            foreach (var outputMetadata in _session.OutputMetadata)
            {
                var outputName = outputMetadata.Key;
                var nodeMetadata = outputMetadata.Value;
                
                // Create pre-allocated tensor for each output based on type
                if (nodeMetadata.IsTensor && nodeMetadata.ElementType == typeof(float))
                {
                    CreatePreallocatedTensor<float>(outputName, nodeMetadata.Dimensions);
                }
                else if (nodeMetadata.IsTensor && nodeMetadata.ElementType == typeof(long))
                {
                    CreatePreallocatedTensor<long>(outputName, nodeMetadata.Dimensions);
                }
                else if (nodeMetadata.IsTensor && nodeMetadata.ElementType == typeof(int))
                {
                    CreatePreallocatedTensor<int>(outputName, nodeMetadata.Dimensions);
                }
                // Additional type handling can be added here as needed
            }
            
            return _session;
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Disposes of the Model instance.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes of the Model instance.
        /// </summary>
        /// <param name="disposing">True if the Dispose method was called explicitly, false if the object is being finalized</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    _session?.Dispose();
                    _inputs?.Clear();
                    _preallocatedOutputs?.Clear();
                }
                // Release unmanaged resources.
                // Set large fields to null.                
                _disposed = true;
            }
        }
        
        /// <summary>
        /// Finalizer for the Model class.
        /// </summary>
        ~Model()
        {
            Dispose(false);
        }

        #endregion
    }
}
