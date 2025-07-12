using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace LiveTalk.Utils
{
    /// <summary>
    /// File system utilities for LiveTalk frame loading and processing.
    /// Provides comprehensive functionality for discovering, loading, and managing driving frame sequences
    /// from Unity's StreamingAssets directory with support for common image formats and robust error handling.
    /// All methods are optimized for Unity's file system patterns and cross-platform compatibility.
    /// </summary>
    internal static class FileUtils
    {
        #region Constants

        /// <summary>
        /// Supported image file extensions for frame loading.
        /// These formats are commonly used for driving frame sequences and provide
        /// broad compatibility across different image sources and platforms.
        /// </summary>
        private static readonly string[] SupportedExtensions = { ".png", ".jpg", ".jpeg" };

        #endregion

        #region Public Methods - Frame Discovery

        /// <summary>
        /// Discovers and returns file paths for driving frame images in the specified folder.
        /// This method scans the StreamingAssets directory for supported image formats and returns
        /// sorted file paths for consistent frame sequence loading. Useful for frame counting and validation.
        /// </summary>
        /// <param name="framesFolderPath">The relative path from StreamingAssets to the frames folder</param>
        /// <param name="maxFrames">Maximum number of frame files to return (-1 for all files)</param>
        /// <returns>Array of full file paths to frame images, sorted by filename for consistent ordering</returns>
        /// <exception cref="ArgumentNullException">Thrown when framesFolderPath is null</exception>
        /// <exception cref="System.IO.DirectoryNotFoundException">Thrown when the specified directory does not exist</exception>
        public static string[] GetFrameFiles(string framesFolderPath, int maxFrames = -1)
        {
            if (string.IsNullOrEmpty(framesFolderPath))
                throw new ArgumentNullException(nameof(framesFolderPath), "Frames folder path cannot be null or empty");

            string fullFolderPath = System.IO.Path.Combine(Application.streamingAssetsPath, framesFolderPath);
            
            if (!System.IO.Directory.Exists(fullFolderPath))
            {
                return new string[0]; // Return empty array instead of throwing exception for graceful handling
            }

            var allFiles = new List<string>();
            
            // Discover all supported image files in the directory
            foreach (string extension in SupportedExtensions)
            {
                string[] files = System.IO.Directory.GetFiles(fullFolderPath, "*" + extension, System.IO.SearchOption.TopDirectoryOnly);
                allFiles.AddRange(files);
            }

            // Sort files by filename for consistent ordering (handles numbered sequences like 00000000, 00000001, etc.)
            allFiles.Sort((a, b) => string.Compare(System.IO.Path.GetFileNameWithoutExtension(a), 
                                                  System.IO.Path.GetFileNameWithoutExtension(b), 
                                                  System.StringComparison.Ordinal));

            // Apply frame limit if specified
            if (maxFrames > 0)
            {
                allFiles = allFiles.Take(maxFrames).ToList();
            }

            return allFiles.ToArray();
        }

        #endregion

        #region Public Methods - Frame Loading

        /// <summary>
        /// Loads and converts driving frame images from the specified folder into Unity Texture2D objects.
        /// This method handles the complete pipeline of file discovery, image loading, format conversion,
        /// and error recovery to provide a robust frame loading solution for driving frame sequences.
        /// All loaded textures are converted to RGB24 format for consistent processing.
        /// </summary>
        /// <param name="drivingFramesFolderPath">The relative path from StreamingAssets to the driving frames folder</param>
        /// <returns>A list of loaded Texture2D objects sorted by filename, or null if no valid frames are found</returns>
        /// <exception cref="ArgumentException">Thrown when drivingFramesFolderPath is null or empty</exception>
        public static List<Texture2D> LoadFramesFromFolder(string drivingFramesFolderPath)
        {
            // Validate input parameters
            if (string.IsNullOrEmpty(drivingFramesFolderPath))
            {
                return null;
            }

            string fullFolderPath = System.IO.Path.Combine(Application.streamingAssetsPath, drivingFramesFolderPath);
            
            // Verify folder exists
            if (!System.IO.Directory.Exists(fullFolderPath))
            {
                Logger.LogWarning($"[FileUtils] Driving frames folder not found: {fullFolderPath}");
                return null;
            }

            var framesList = new List<Texture2D>();
            
            // Load all supported image files from the folder
            foreach (string extension in SupportedExtensions)
            {
                string[] files = System.IO.Directory.GetFiles(fullFolderPath, "*" + extension, System.IO.SearchOption.TopDirectoryOnly);
                
                foreach (string filePath in files)
                {
                    Texture2D loadedTexture = LoadFrame(filePath);
                    if (loadedTexture != null)
                    {
                        framesList.Add(loadedTexture);
                    }
                }
            }
            
            // Validate that frames were loaded successfully
            if (framesList.Count == 0)
            {
                Logger.LogWarning($"[FileUtils] No valid images found in driving frames folder: {fullFolderPath}");
                return null;
            }

            // Sort frames by name for consistent sequence ordering
            framesList.Sort((a, b) => string.Compare(a.name, b.name, System.StringComparison.Ordinal));
            
            Logger.Log($"[FileUtils] Successfully loaded {framesList.Count} driving frames from: {drivingFramesFolderPath}");
            return framesList;
        }

        /// <summary>
        /// Loads a single image file and converts it to RGB24 format for processing.
        /// This method handles file reading, texture creation, format conversion, and error recovery
        /// for individual frame images with comprehensive logging for debugging.
        /// </summary>
        /// <param name="filePath">The full file path to the image file to load</param>
        /// <returns>A Texture2D in RGB24 format with the filename as the texture name, or null if loading fails</returns>
        public static Texture2D LoadFrame(string filePath)
        {
            try
            {
                // Read raw image data from file
                byte[] fileData = System.IO.File.ReadAllBytes(filePath);
                
                // Create temporary texture for loading (size will be adjusted by LoadImage)
                Texture2D texture = new(2, 2);
                
                // Load image data into texture
                if (texture.LoadImage(fileData))
                {
                    // Set texture name to filename for identification
                    texture.name = System.IO.Path.GetFileNameWithoutExtension(filePath);
                    
                    // Convert to RGB24 format for consistent processing
                    Texture2D convertedTexture = TextureUtils.ConvertTexture2DToRGB24(texture);
                    
                    // Clean up temporary texture if conversion created a new one
                    if (convertedTexture != texture)
                    {
                        UnityEngine.Object.DestroyImmediate(texture);
                    }
                    
                    return convertedTexture;
                }
                else
                {
                    Logger.LogWarning($"[FileUtils] Failed to load image data for: {filePath}");
                    UnityEngine.Object.DestroyImmediate(texture);
                    return null;
                }
            }
            catch (System.IO.IOException ioEx)
            {
                Logger.LogError($"[FileUtils] IO error loading driving frame {filePath}: {ioEx.Message}");
                return null;
            }
            catch (UnauthorizedAccessException accessEx)
            {
                Logger.LogError($"[FileUtils] Access denied loading driving frame {filePath}: {accessEx.Message}");
                return null;
            }
            catch (Exception ex)
            {
                Logger.LogError($"[FileUtils] Unexpected error loading driving frame {filePath}: {ex.Message}");
                return null;
            }
        }

        #endregion
    }
}
