using System;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace LiveTalk.Utils
{
    using Core;
    
    /// <summary>
    /// Comprehensive texture processing utilities for LiveTalk image and frame operations.
    /// Provides high-performance texture format conversion, Unity integration, and optimized 
    /// coordinate system transformations with unsafe memory operations for maximum performance.
    /// All methods handle Unity's bottom-left texture coordinate system and cross-platform compatibility.
    /// </summary>
    internal static class TextureUtils
    {
        #region Public Methods - Unity Texture Operations
        
        /// <summary>
        /// Creates a readable copy of a texture using RenderTexture blitting for cross-platform compatibility.
        /// This method handles compressed and non-readable textures by rendering them to a RenderTexture
        /// and then reading the pixels back to create a readable RGB24 texture.
        /// </summary>
        /// <param name="source">The source texture to make readable (can be compressed or non-readable)</param>
        /// <returns>A new readable Texture2D in RGB24 format with the same dimensions as the source</returns>
        /// <exception cref="ArgumentNullException">Thrown when source is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when RenderTexture operations fail</exception>
        public static Texture2D MakeTextureReadable(Texture2D source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            
            // Create RenderTexture for blitting operation
            RenderTexture renderTexture = RenderTexture.GetTemporary(
                source.width, 
                source.height, 
                0, 
                RenderTextureFormat.ARGB32
            );
            
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = renderTexture;
            
            try
            {
                // Blit source texture to RenderTexture (handles compressed formats)
                Graphics.Blit(source, renderTexture);
                
                // Create new readable texture and read pixels from RenderTexture
                Texture2D readableTexture = new(source.width, source.height, TextureFormat.RGB24, false);
                readableTexture.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
                readableTexture.Apply();
                
                return readableTexture;
            }
            finally
            {
                // Clean up RenderTexture resources
                RenderTexture.active = previousActive;
                RenderTexture.ReleaseTemporary(renderTexture);
            }
        }
        
        /// <summary>
        /// Resizes a texture to specified dimensions using Unity's RenderTexture system for high-quality scaling.
        /// This method uses bilinear filtering for smooth scaling and handles both upscaling and downscaling operations.
        /// </summary>
        /// <param name="source">The source texture to resize</param>
        /// <param name="targetWidth">The target width in pixels (must be positive)</param>
        /// <param name="targetHeight">The target height in pixels (must be positive)</param>
        /// <returns>A new resized Texture2D in RGB24 format with the specified dimensions</returns>
        /// <exception cref="ArgumentNullException">Thrown when source is null</exception>
        /// <exception cref="ArgumentException">Thrown when target dimensions are not positive</exception>
        /// <exception cref="InvalidOperationException">Thrown when RenderTexture operations fail</exception>
        public static Texture2D ResizeTexture(Texture2D source, int targetWidth, int targetHeight)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (targetWidth <= 0)
                throw new ArgumentException("Target width must be positive", nameof(targetWidth));
            if (targetHeight <= 0)
                throw new ArgumentException("Target height must be positive", nameof(targetHeight));
                
            // Create RenderTexture for scaling operation
            RenderTexture renderTexture = RenderTexture.GetTemporary(targetWidth, targetHeight, 0, RenderTextureFormat.ARGB32);
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = renderTexture;
            
            try
            {
                // Scale the source texture using bilinear filtering
                Graphics.Blit(source, renderTexture);
                
                // Create new texture from scaled RenderTexture
                Texture2D resizedTexture = new(targetWidth, targetHeight, TextureFormat.RGB24, false);
                resizedTexture.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
                resizedTexture.Apply();
                
                return resizedTexture;
            }
            finally
            {
                // Clean up RenderTexture resources
                RenderTexture.active = previousActive;
                RenderTexture.ReleaseTemporary(renderTexture);
            }
        }

        #endregion

        #region Public Methods - Format Conversion
        
        /// <summary>
        /// Converts a Texture2D to RGB24 format for consistent processing across the LiveTalk pipeline.
        /// This method ensures all textures are in a standardized format suitable for neural network inference.
        /// If the texture is already in RGB24 format, it returns the original texture without conversion.
        /// </summary>
        /// <param name="texture">The texture to convert to RGB24 format</param>
        /// <returns>A Texture2D in RGB24 format (either the original or a new converted texture)</returns>
        /// <exception cref="ArgumentNullException">Thrown when texture is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when pixel data conversion fails</exception>
        public static Texture2D ConvertTexture2DToRGB24(Texture2D texture)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));
            
            if (texture.format != TextureFormat.RGB24)
            {
                var convertedTexture = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false)
                {
                    name = texture.name
                };
                convertedTexture.SetPixels(texture.GetPixels());
                convertedTexture.Apply();
                return convertedTexture;
            }
            return texture;
        }
        
        /// <summary>
        /// Converts a compressed texture to an uncompressed format suitable for PNG encoding and processing.
        /// This method handles various compressed texture formats and provides fallback mechanisms for compatibility.
        /// Uses RenderTexture blitting for maximum compatibility across different platforms and texture formats.
        /// </summary>
        /// <param name="source">The source texture to convert (can be in any format)</param>
        /// <returns>A new uncompressed Texture2D in RGBA32 format suitable for PNG encoding</returns>
        /// <exception cref="ArgumentNullException">Thrown when source is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when conversion fails for unsupported formats</exception>
        public static Texture2D ConvertToUncompressedTexture(Texture2D source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            // Check if texture is already in a format that supports EncodeToPNG
            if (source.format == TextureFormat.RGBA32 || 
                source.format == TextureFormat.RGB24 || 
                source.format == TextureFormat.ARGB32)
            {
                return source; // Already uncompressed
            }

            // Create a new uncompressed texture
            Texture2D uncompressed = new(source.width, source.height, TextureFormat.RGBA32, false);
            
            // Use Graphics.CopyTexture if possible for better performance
            if (SystemInfo.copyTextureSupport != UnityEngine.Rendering.CopyTextureSupport.None)
            {
                try
                {
                    // Create a temporary RenderTexture and blit for format conversion
                    RenderTexture tempRT = RenderTexture.GetTemporary(source.width, source.height, 0, RenderTextureFormat.ARGB32);
                    Graphics.Blit(source, tempRT);
                    
                    // Read from RenderTexture to uncompressed texture
                    RenderTexture.active = tempRT;
                    uncompressed.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
                    uncompressed.Apply();
                    RenderTexture.active = null;
                    
                    // Clean up RenderTexture resources
                    RenderTexture.ReleaseTemporary(tempRT);
                    
                    return uncompressed;
                }
                catch (Exception ex)
                {
                    Logger.LogWarning($"[TextureUtils] Graphics.Blit failed, falling back to pixel copy: {ex.Message}");
                }
            }
            
            // Fallback: try to get pixels directly (may not work with all compressed formats)
            try
            {
                Color[] pixels = source.GetPixels();
                uncompressed.SetPixels(pixels);
                uncompressed.Apply();
                return uncompressed;
            }
            catch (Exception ex)
            {
                Logger.LogError($"[TextureUtils] Failed to convert texture format: {ex.Message}");
                UnityEngine.Object.DestroyImmediate(uncompressed);
                throw new InvalidOperationException($"Cannot convert texture format {source.format} to uncompressed format for PNG encoding. Please use an uncompressed texture format.", ex);
            }
        }

        #endregion

        #region Public Methods - Frame Conversion
        
        /// <summary>
        /// Converts a Unity Texture2D to a Frame structure with RGB24 byte array data and coordinate system transformation.
        /// This method handles Unity's bottom-left coordinate system by flipping the Y-axis to match standard top-left
        /// image coordinate systems used in computer vision and neural network processing.
        /// Uses unsafe memory operations and parallel processing for maximum performance.
        /// </summary>
        /// <param name="img">The Unity Texture2D to convert (must be in RGB24 format)</param>
        /// <returns>A Frame structure containing RGB24 byte array data with top-left coordinate system</returns>
        /// <exception cref="ArgumentNullException">Thrown when img is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when texture format is not supported or pixel data access fails</exception>
        public static unsafe Frame Texture2DToFrame(Texture2D img)
        {
            if (img == null)
                throw new ArgumentNullException(nameof(img));
            
            int h = img.height;
            int w = img.width;
            int rowBytes = w * 3; // RGB24 = 3 bytes per pixel
            
            // Get initial image data directly from texture (assumes RGB24 format)
            var pixelData = img.GetPixelData<byte>(0);
            var imageData = new byte[pixelData.Length];
            
            byte* srcPtr = (byte*)pixelData.GetUnsafeReadOnlyPtr();
            
            fixed (byte* dstPtr = imageData)
            {
                byte* srcPtrLocal = srcPtr;
                byte* dstPtrLocal = dstPtr;
                
                // Parallel processing with Y-axis flipping for coordinate system transformation
                System.Threading.Tasks.Parallel.For(0, h, y =>
                {
                    byte* srcRowPtr = srcPtrLocal + (h - 1 - y) * rowBytes; // Source row (Unity bottom-left, flipped)
                    byte* dstRowPtr = dstPtrLocal + y * rowBytes;            // Destination row (standard top-left)
                    Buffer.MemoryCopy(srcRowPtr, dstRowPtr, rowBytes, rowBytes);
                });
            }

            return new Frame(imageData, w, h);
        }
        
        /// <summary>
        /// Converts a Frame structure with RGB24 byte array data back to a Unity Texture2D with coordinate system transformation.
        /// This method handles standard top-left coordinate system by flipping the Y-axis to match Unity's bottom-left
        /// texture coordinate system. Uses unsafe memory operations and parallel processing for maximum performance.
        /// </summary>
        /// <param name="frame">The Frame structure containing RGB24 byte array data with top-left coordinate system</param>
        /// <returns>A Unity Texture2D in RGB24 format with Unity's bottom-left coordinate system</returns>
        /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
        /// <exception cref="ArgumentException">Thrown when frame data is null or dimensions are invalid</exception>
        /// <exception cref="InvalidOperationException">Thrown when texture creation or pixel data access fails</exception>
        public static unsafe Texture2D FrameToTexture2D(Frame frame)
        {
            if (frame.data == null)
                throw new ArgumentException("Frame data cannot be null", nameof(frame));
            if (frame.width <= 0 || frame.height <= 0)
                throw new ArgumentException("Frame dimensions must be positive", nameof(frame));
            
            var texture = new Texture2D(frame.width, frame.height, TextureFormat.RGB24, false);
            
            var pixelData = texture.GetPixelData<byte>(0);
            byte* texturePtr = (byte*)pixelData.GetUnsafePtr();
            
            // OPTIMIZED: Process with unsafe pointers and parallelization
            fixed (byte* imagePtrFixed = frame.data)
            {
                // Capture pointer in local variable to avoid lambda closure issues
                byte* imagePtrLocal = imagePtrFixed;
                
                // Parallel processing with Y-axis flipping for coordinate system transformation
                System.Threading.Tasks.Parallel.For(0, frame.height, y =>
                {
                    // Calculate Unity texture coordinate (bottom-left origin) from image coordinate (top-left origin)
                    int unityY = frame.height - 1 - y; // Flip Y coordinate for Unity coordinate system
                    
                    // Calculate row pointers using direct pointer arithmetic
                    byte* srcRowPtr = imagePtrLocal + y * frame.width * 3;        // Source row (standard top-left)
                    byte* dstRowPtr = texturePtr + unityY * frame.width * 3;      // Destination row (Unity bottom-left)
                    
                    int rowBytes = frame.width * 3; // RGB24 = 3 bytes per pixel
                    Buffer.MemoryCopy(srcRowPtr, dstRowPtr, rowBytes, rowBytes);
                });
            }
            
            // Apply changes to texture (no need for SetPixels since we wrote directly to pixel data)
            texture.Apply();
            return texture;
        }

        #endregion

        #region Private Methods - Helper Operations
        
        /// <summary>
        /// Checks if a texture format is compressed and requires conversion for direct pixel access.
        /// This method identifies common compressed formats across different platforms including DXT, ETC, ASTC, and BC formats.
        /// </summary>
        /// <param name="format">The texture format to check</param>
        /// <returns>True if the format is compressed and requires conversion; false if it allows direct pixel access</returns>
        private static bool IsCompressedFormat(TextureFormat format)
        {
            return format == TextureFormat.DXT1 ||
                   format == TextureFormat.DXT5 ||
                   format == TextureFormat.BC4 ||
                   format == TextureFormat.BC5 ||
                   format == TextureFormat.BC6H ||
                   format == TextureFormat.BC7 ||
                   format == TextureFormat.ETC_RGB4 ||
                   format == TextureFormat.ETC2_RGBA8 ||
                   format == TextureFormat.ASTC_4x4 ||
                   format == TextureFormat.ASTC_5x5 ||
                   format == TextureFormat.ASTC_6x6 ||
                   format == TextureFormat.ASTC_8x8 ||
                   format == TextureFormat.ASTC_10x10 ||
                   format == TextureFormat.ASTC_12x12;
        }

        #endregion
    }
}
