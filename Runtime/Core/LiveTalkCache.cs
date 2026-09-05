using System;
using System.IO;
using UnityEngine;

namespace LiveTalk.Core
{
    using Utils;

    /// <summary>
    /// Internal caching system for LiveTalk.
    /// Provides persistent caching for speech audio and other generated content.
    /// </summary>
    internal static class LiveTalkCache
    {
        private static bool _enabled = true;
        private static string _path;
        private static bool _initialized = false;

        /// <summary>
        /// Gets whether the cache is enabled and initialized.
        /// </summary>
        public static bool IsEnabled => _enabled && _initialized;

        /// <summary>
        /// Gets the path to the cache directory.
        /// </summary>
        public static string Path => _path;

        /// <summary>
        /// Gets whether the cache has been initialized.
        /// </summary>
        public static bool IsInitialized => _initialized;

        /// <summary>
        /// Initialize the cache with the specified path.
        /// </summary>
        /// <param name="cachePath">Path to store cached files</param>
        /// <param name="enabled">Whether caching is enabled</param>
        internal static void Initialize(string cachePath, bool enabled = true)
        {
            _enabled = enabled;
            _path = cachePath;

            if (!string.IsNullOrEmpty(cachePath) && enabled)
            {
                if (!Directory.Exists(cachePath))
                {
                    Directory.CreateDirectory(cachePath);
                }
                _initialized = true;
                Logger.Log($"[Cache] Initialized at: {cachePath}");
            }
            else
            {
                _initialized = false;
                Logger.Log($"[Cache] Disabled");
            }
        }

        /// <summary>
        /// Enable or disable the cache at runtime.
        /// </summary>
        /// <param name="enabled">Whether to enable caching</param>
        internal static void SetEnabled(bool enabled)
        {
            _enabled = enabled;
            Logger.Log($"[Cache] {(enabled ? "Enabled" : "Disabled")}");
        }

        /// <summary>
        /// Get the full file path for a given cache key and extension.
        /// </summary>
        /// <param name="cacheKey">The unique cache key (from HashUtils)</param>
        /// <param name="extension">File extension including dot (default: .wav)</param>
        /// <returns>Full path to the cached file, or null if cache not initialized</returns>
        public static string GetFilePath(string cacheKey, string extension = ".wav")
        {
            if (!IsEnabled || string.IsNullOrEmpty(cacheKey))
                return null;
            
            return System.IO.Path.Combine(_path, $"{cacheKey}{extension}");
        }

        /// <summary>
        /// Check if a cached file exists for the given cache key.
        /// </summary>
        /// <param name="cacheKey">The unique cache key</param>
        /// <param name="extension">File extension including dot (default: .wav)</param>
        /// <returns>Tuple of (exists, filePath)</returns>
        public static (bool exists, string filePath) CheckExists(string cacheKey, string extension = ".wav")
        {
            if (!IsEnabled || string.IsNullOrEmpty(cacheKey))
                return (false, null);

            string filePath = GetFilePath(cacheKey, extension);
            if (File.Exists(filePath))
                return (true, filePath);

            return (false, null);
        }

        /// <summary>
        /// Clear all cached content (files and folders) in the initialized cache.
        /// No-op before <see cref="Initialize"/>; use <see cref="Clear(string)"/>
        /// with an explicit location then.
        /// </summary>
        public static void Clear()
        {
            if (!_initialized || string.IsNullOrEmpty(_path))
                return;
            Clear(_path);
        }

        /// <summary>
        /// Clear all cached content under an explicit cache location. Works
        /// before <see cref="Initialize"/>, so a host can offer "clear cache"
        /// without first paying for model initialization. The folder itself
        /// is kept.
        /// </summary>
        /// <param name="cachePath">The cache folder to empty</param>
        public static void Clear(string cachePath)
        {
            if (string.IsNullOrEmpty(cachePath))
                return;

            try
            {
                if (Directory.Exists(cachePath))
                {
                    int filesCleared = 0;
                    int foldersCleared = 0;
                    
                    // Clear all files
                    var files = Directory.GetFiles(cachePath);
                    foreach (var file in files)
                    {
                        File.Delete(file);
                        filesCleared++;
                    }
                    
                    // Clear all subdirectories (per-utterance frame folders)
                    var directories = Directory.GetDirectories(cachePath);
                    foreach (var dir in directories)
                    {
                        Directory.Delete(dir, true);
                        foldersCleared++;
                    }
                    
                    Logger.Log($"[Cache] Cleared {filesCleared} files and {foldersCleared} folders");
                }
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Cache] Error clearing cache: {ex.Message}");
            }
        }

        /// <summary>
        /// Get the total size of all cached content in bytes (files and folders)
        /// in the initialized cache. 0 before <see cref="Initialize"/>; use
        /// <see cref="GetSize(string)"/> with an explicit location then.
        /// </summary>
        public static long GetSize()
        {
            if (!_initialized || string.IsNullOrEmpty(_path))
                return 0;
            return GetSize(_path);
        }

        /// <summary>
        /// Get the total size in bytes of everything under an explicit cache
        /// location. Works before <see cref="Initialize"/>. 0 when the folder
        /// does not exist or cannot be read.
        /// </summary>
        /// <param name="cachePath">The cache folder to measure</param>
        public static long GetSize(string cachePath)
        {
            if (string.IsNullOrEmpty(cachePath) || !Directory.Exists(cachePath))
                return 0;

            try
            {
                return GetDirectorySize(cachePath);
            }
            catch
            {
                return 0;
            }
        }

        /// <summary>
        /// Recursively calculate directory size.
        /// </summary>
        private static long GetDirectorySize(string path)
        {
            long size = 0;
            
            // Add file sizes
            foreach (var file in Directory.GetFiles(path))
            {
                size += new FileInfo(file).Length;
            }
            
            // Add subdirectory sizes
            foreach (var dir in Directory.GetDirectories(path))
            {
                size += GetDirectorySize(dir);
            }
            
            return size;
        }

        #region Animation Frame Caching

        /// <summary>
        /// Get the folder path for cached animation frames.
        /// The frames key (from <c>HashUtils.GenerateFramesCacheKey</c>) covers
        /// voice, text, avatar and expression; the folder is that key with a
        /// "_frames" suffix.
        /// </summary>
        /// <param name="speechCacheKey">The frames cache key</param>
        /// <returns>Full path to the frames folder, or null if cache not initialized</returns>
        public static string GetFramesFolderPath(string speechCacheKey)
        {
            if (!IsEnabled || string.IsNullOrEmpty(speechCacheKey))
                return null;
            
            return System.IO.Path.Combine(_path, $"{speechCacheKey}_frames");
        }

        /// <summary>
        /// Check if cached animation frames exist for the given frames cache key.
        /// Validates that all expected frames are present.
        /// </summary>
        /// <param name="speechCacheKey">The frames cache key</param>
        /// <param name="expectedFrameCount">Expected number of frames (0 means check existence only)</param>
        /// <returns>Tuple of (exists, folderPath, actualFrameCount)</returns>
        public static (bool exists, string folderPath, int frameCount) CheckFramesCacheExists(
            string speechCacheKey, 
            int expectedFrameCount = 0)
        {
            if (!IsEnabled || string.IsNullOrEmpty(speechCacheKey))
                return (false, null, 0);

            string framesFolder = GetFramesFolderPath(speechCacheKey);
            if (!Directory.Exists(framesFolder))
                return (false, framesFolder, 0);

            // Count PNG files in the folder
            var pngFiles = Directory.GetFiles(framesFolder, "frame_*.png");
            int frameCount = pngFiles.Length;

            if (frameCount == 0)
                return (false, framesFolder, 0);

            // If expectedFrameCount is specified, check if we have all frames
            if (expectedFrameCount > 0 && frameCount != expectedFrameCount)
            {
                Logger.LogVerbose($"[Cache] Frame cache incomplete: found {frameCount}/{expectedFrameCount} frames");
                return (false, framesFolder, frameCount);
            }

            return (true, framesFolder, frameCount);
        }

        /// <summary>
        /// Get the file path for a specific cached frame.
        /// </summary>
        /// <param name="speechCacheKey">The speech cache key</param>
        /// <param name="frameIndex">The frame index (0-based)</param>
        /// <returns>Full path to the frame file</returns>
        public static string GetFramePath(string speechCacheKey, int frameIndex)
        {
            string framesFolder = GetFramesFolderPath(speechCacheKey);
            if (string.IsNullOrEmpty(framesFolder))
                return null;
            
            return System.IO.Path.Combine(framesFolder, $"frame_{frameIndex:D6}.png");
        }

        /// <summary>
        /// Create the frames cache folder for storing animation frames.
        /// </summary>
        /// <param name="speechCacheKey">The speech cache key</param>
        /// <returns>The created folder path, or null if failed</returns>
        public static string CreateFramesCacheFolder(string speechCacheKey)
        {
            if (!IsEnabled || string.IsNullOrEmpty(speechCacheKey))
                return null;

            try
            {
                string framesFolder = GetFramesFolderPath(speechCacheKey);
                if (!Directory.Exists(framesFolder))
                {
                    Directory.CreateDirectory(framesFolder);
                }
                return framesFolder;
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Cache] Error creating frames cache folder: {ex.Message}");
                return null;
            }
        }

        /// <summary>Name of the file inside a frames folder recording the avatar frame the entry was rendered from.</summary>
        private const string FramesStartFileName = "start.txt";

        /// <summary>
        /// Records the avatar frame index the first cached frame was rendered
        /// onto, so a replay can line the idle loop up to it. Best effort: a
        /// failure to write leaves the entry readable as "from frame 0".
        /// </summary>
        public static void WriteFramesStartIndex(string framesFolder, int startFrameIndex)
        {
            if (string.IsNullOrEmpty(framesFolder))
                return;
            try
            {
                File.WriteAllText(System.IO.Path.Combine(framesFolder, FramesStartFileName),
                    startFrameIndex.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }
            catch (Exception ex)
            {
                Logger.LogWarning($"[Cache] Could not record the start frame of {framesFolder}: {ex.Message}");
            }
        }

        /// <summary>
        /// The avatar frame index a cached frames entry was rendered from, or 0
        /// when the entry predates the record (every such entry started at 0).
        /// </summary>
        public static int ReadFramesStartIndex(string framesFolder)
        {
            if (string.IsNullOrEmpty(framesFolder))
                return 0;
            try
            {
                string path = System.IO.Path.Combine(framesFolder, FramesStartFileName);
                if (File.Exists(path)
                    && int.TryParse(File.ReadAllText(path).Trim(), System.Globalization.NumberStyles.Integer,
                        System.Globalization.CultureInfo.InvariantCulture, out int start))
                    return Math.Max(0, start);
            }
            catch (Exception ex)
            {
                Logger.LogWarning($"[Cache] Could not read the start frame of {framesFolder}: {ex.Message}");
            }
            return 0;
        }

        /// <summary>
        /// Delete cached animation frames for a given speech cache key.
        /// </summary>
        /// <param name="speechCacheKey">The speech cache key</param>
        public static void DeleteFramesCache(string speechCacheKey)
        {
            if (!IsEnabled || string.IsNullOrEmpty(speechCacheKey))
                return;

            try
            {
                string framesFolder = GetFramesFolderPath(speechCacheKey);
                if (Directory.Exists(framesFolder))
                {
                    Directory.Delete(framesFolder, true);
                    Logger.LogVerbose($"[Cache] Deleted frames cache: {speechCacheKey}");
                }
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Cache] Error deleting frames cache: {ex.Message}");
            }
        }

        #endregion
    }
}

