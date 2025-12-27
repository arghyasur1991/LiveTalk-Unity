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
        /// Get the folder path for a given cache key (for caching folder-based content like voice data).
        /// </summary>
        /// <param name="cacheKey">The unique cache key</param>
        /// <returns>Full path to the cached folder, or null if cache not initialized</returns>
        public static string GetFolderPath(string cacheKey)
        {
            if (!IsEnabled || string.IsNullOrEmpty(cacheKey))
                return null;
            
            return System.IO.Path.Combine(_path, cacheKey);
        }

        /// <summary>
        /// Check if a cached folder exists for the given cache key.
        /// </summary>
        /// <param name="cacheKey">The unique cache key</param>
        /// <returns>Tuple of (exists, folderPath)</returns>
        public static (bool exists, string folderPath) CheckFolderExists(string cacheKey)
        {
            if (!IsEnabled || string.IsNullOrEmpty(cacheKey))
                return (false, null);

            string folderPath = GetFolderPath(cacheKey);
            if (Directory.Exists(folderPath) && Directory.GetFiles(folderPath).Length > 0)
                return (true, folderPath);

            return (false, null);
        }

        /// <summary>
        /// Recursively copy all files and subdirectories from source folder to destination folder.
        /// </summary>
        /// <param name="sourceFolder">Source folder path</param>
        /// <param name="destFolder">Destination folder path</param>
        public static void CopyFolder(string sourceFolder, string destFolder)
        {
            if (!Directory.Exists(sourceFolder))
                return;

            if (!Directory.Exists(destFolder))
                Directory.CreateDirectory(destFolder);

            // Copy all files
            foreach (var file in Directory.GetFiles(sourceFolder))
            {
                string fileName = System.IO.Path.GetFileName(file);
                string destFile = System.IO.Path.Combine(destFolder, fileName);
                File.Copy(file, destFile, true);
            }

            // Recursively copy all subdirectories
            foreach (var dir in Directory.GetDirectories(sourceFolder))
            {
                string dirName = System.IO.Path.GetFileName(dir);
                string destSubDir = System.IO.Path.Combine(destFolder, dirName);
                CopyFolder(dir, destSubDir);
            }
        }

        /// <summary>
        /// Clear all cached content (files and folders).
        /// </summary>
        public static void Clear()
        {
            if (!_initialized || string.IsNullOrEmpty(_path))
                return;

            try
            {
                if (Directory.Exists(_path))
                {
                    int filesCleared = 0;
                    int foldersCleared = 0;
                    
                    // Clear all files
                    var files = Directory.GetFiles(_path);
                    foreach (var file in files)
                    {
                        File.Delete(file);
                        filesCleared++;
                    }
                    
                    // Clear all subdirectories (voice sample folders, etc.)
                    var directories = Directory.GetDirectories(_path);
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
        /// Get the total size of all cached content in bytes (files and folders).
        /// </summary>
        public static long GetSize()
        {
            if (!_initialized || string.IsNullOrEmpty(_path) || !Directory.Exists(_path))
                return 0;

            try
            {
                return GetDirectorySize(_path);
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
        /// Uses the same cache key as speech with "_frames" suffix.
        /// </summary>
        /// <param name="speechCacheKey">The speech cache key (from GenerateSpeechCacheKey)</param>
        /// <returns>Full path to the frames folder, or null if cache not initialized</returns>
        public static string GetFramesFolderPath(string speechCacheKey)
        {
            if (!IsEnabled || string.IsNullOrEmpty(speechCacheKey))
                return null;
            
            return System.IO.Path.Combine(_path, $"{speechCacheKey}_frames");
        }

        /// <summary>
        /// Check if cached animation frames exist for the given speech cache key.
        /// Validates that all expected frames are present.
        /// </summary>
        /// <param name="speechCacheKey">The speech cache key</param>
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

