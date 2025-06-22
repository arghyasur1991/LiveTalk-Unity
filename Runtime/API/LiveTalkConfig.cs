using System;

namespace LiveTalk.API
{
    /// <summary>
    /// Configuration for MuseTalk inference
    /// </summary>
    [Serializable]
    public class LiveTalkConfig
    {
        public string ModelPath = "LiveTalk";
        public string Version = "v15"; // only v15 is supported
        public string Device = "cpu"; // "cpu" or "cuda"
        public int BatchSize = 4;
        public float ExtraMargin { get; internal set; } = 10f; // Additional margin for v15
        public bool UseINT8 { get; internal set; } = false; // Enable INT8 quantization TODO: Int8 is not currently supported. Keeping it false for now.
        
        // Disk caching configuration
        public bool EnableDiskCache = false; // Enable persistent disk caching for avatar processing
        public string CacheDirectory = ""; // Cache directory path (empty = auto-detect)
        public int MaxCacheEntriesPerAvatar = 1000; // Maximum cache entries per avatar hash
        public long MaxCacheSizeMB = 1024; // Maximum total cache size in MB (1GB default)
        public int CacheVersionNumber { get; internal set; } = 1; // Cache version for invalidation on format changes
        public bool CacheLatentsOnly = false; // Cache only latents (faster) vs full avatar data (slower but complete)
        
        public LiveTalkConfig()
        {
        }
        
        public LiveTalkConfig(string modelPath, string version = "v15")
        {
            if (version != "v15")
            {
                throw new NotSupportedException("Only v15 is supported");
            }
            ModelPath = modelPath;
            Version = version;
        }
    }
}