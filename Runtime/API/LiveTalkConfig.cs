using System;
using System.IO;

namespace LiveTalk.API
{
    /// <summary>
    /// Configuration for MuseTalk inference
    /// </summary>
    [Serializable]
    public class LiveTalkConfig
    {
        public string ModelPath = "";
        public string Version = "v15"; // only v15 is supported. Not used for now.
        public string Device = "cpu"; // "cpu" or "cuda"
        public float ExtraMargin { get; internal set; } = 10f; // Additional margin for v15
        // Disk caching configuration
        public bool EnableDiskCache = false; // Enable persistent disk caching for avatar processing
        public string CacheDirectory = ""; // Cache directory path (empty = auto-detect)
        public int MaxCacheEntriesPerAvatar = 1000; // Maximum cache entries per avatar hash
        public long MaxCacheSizeMB = 1024; // Maximum total cache size in MB (1GB default)
        public int CacheVersionNumber { get; internal set; } = 1; // Cache version for invalidation on format changes
        public bool CacheLatentsOnly = false; // Cache only latents (faster) vs full avatar data (slower but complete)
        
        public LiveTalkConfig(string modelPath)
        {
            ModelPath = Path.Combine(modelPath, "LiveTalk", "models");
        }
    }
}