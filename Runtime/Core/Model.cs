using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LiveTalk.Core
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;
    using API;
    using UnityEngine;
    using Utils;
    
    internal class Model
    {
        private readonly ModelConfig _config;
        private InferenceSession _session;
        private List<string> _inputNames = new();
        private List<NamedOnnxValue> _inputs;
        private readonly Task<InferenceSession> _loadTask;

        public Model(
            LiveTalkConfig config, 
            string modelName, 
            ExecutionProvider preferredExecutionProvider = ExecutionProvider.CPU, 
            Precision precision = Precision.FP32, 
            string version = "")
        {
            _config = new ModelConfig(modelName, preferredExecutionProvider, precision, version);
            _loadTask = LoadModel(config);
        }

        private async Task<InferenceSession> LoadModel(LiveTalkConfig config)
        {
            await Task.Run(() => {
                _session = ModelUtils.LoadModel(config, _config);
            });
            _inputNames = _session.InputMetadata.Keys.ToList();
            _inputs = new List<NamedOnnxValue>(_inputNames.Count);
            return _session;
        }

        public async Task LoadInput(int index, Tensor<float> inputTensor)
        {
            await _loadTask;
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[index], inputTensor));
        }

        public async Task LoadInput(int index, Tensor<long> inputTensor)
        {
            await _loadTask;
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[index], inputTensor));
        }

        public async Task LoadInputs(List<Tensor<float>> inputTensors)
        {
            await _loadTask;
            var inputNames = _session.InputMetadata.Keys.ToArray();
            if (inputTensors.Count != inputNames.Length)
            {
                throw new Exception($"Input tensors count mismatch: {inputTensors.Count} != {inputNames.Length}");
            }
            for (int i = 0; i < inputTensors.Count; i++)
            {
                _inputs.Add(NamedOnnxValue.CreateFromTensor(inputNames[i], inputTensors[i]));
            }
        }

        public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> Run(List<Tensor<float>> inputTensors)
        {
            await _loadTask;
            await LoadInputs(inputTensors);
            return await Run();
        }

        public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> Run()
        {
            await _loadTask;
            var start = System.Diagnostics.Stopwatch.StartNew();
            var results = _session.Run(_inputs);
            var elapsed = start.ElapsedMilliseconds;
            Debug.Log($"[Model] Run {_config.modelName} took {elapsed}ms");
            return results;
        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }
}