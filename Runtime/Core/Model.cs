using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LiveTalk.Core
{
    using System;
    using System.Linq;
    using API;
    using UnityEngine;
    using Utils;
    
    internal class Model
    {
        private readonly ModelConfig _config;
        private readonly InferenceSession _session;
        private readonly List<string> _inputNames = new();
        private readonly List<NamedOnnxValue> _inputs;

        public Model(
            LiveTalkConfig config, 
            string modelName, 
            ExecutionProvider preferredExecutionProvider = ExecutionProvider.CPU, 
            bool isInt8 = false, 
            string version = "")
        {
            _config = new ModelConfig(modelName, preferredExecutionProvider, isInt8, version);
            _session = ModelUtils.LoadModel(config, _config);
            _inputNames = _session.InputMetadata.Keys.ToList();
            _inputs = new List<NamedOnnxValue>(_inputNames.Count);
        }

        public void LoadInput(int index, Tensor<float> inputTensor)
        {
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[index], inputTensor));
        }

        public void LoadInput(int index, Tensor<long> inputTensor)
        {
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[index], inputTensor));
        }

        public void LoadInputs(List<Tensor<float>> inputTensors)
        {
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

        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(List<Tensor<float>> inputTensors)
        {
            LoadInputs(inputTensors);
            return Run();
        }

        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run()
        {
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