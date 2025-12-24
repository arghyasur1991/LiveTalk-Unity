using System;
using System.IO;

namespace LiveTalk.Core
{    
    using API;
    /// <summary>
    /// Configuration for LiveTalk pipeline
    /// </summary>
    [Serializable]
    internal class LiveTalkConfig
    {
        public string ModelPath = "";
        public LogLevel LogLevel { get; set; } = LogLevel.INFO;
        public MemoryUsage MemoryUsage { get; set; } = MemoryUsage.Balanced;
        internal string Version = "v15"; // only v15 is supported. Not used for now.
        internal float ExtraMargin { get; set; } = 10f; // Additional margin for v15
        
        public LiveTalkConfig(string modelPath, LogLevel logLevel, MemoryUsage memoryUsage)
        {
            ModelPath = Path.Combine(modelPath, "LiveTalk", "models");
            LogLevel = logLevel;
            MemoryUsage = memoryUsage;
        }
    }
}