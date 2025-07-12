using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace LiveTalk.Utils
{
    using Core;
    
    /// <summary>
    /// Advanced image blending utilities for seamless face composition and mask-based blending operations.
    /// Provides high-performance implementations of face blending algorithms with optimized unsafe memory operations,
    /// parallel processing, and efficient mask generation for real-time talking head video synthesis.
    /// All methods are designed for maximum performance.
    /// </summary>
    internal static class ImageBlendingHelper
    {
        #region Public Methods - Main Blending Operations

        /// <summary>
        /// Performs seamless face blending with original image using precomputed segmentation masks.
        /// This is the main entry point for face composition with optimized performance using 
        /// cached segmentation data to avoid regenerating masks for every frame.
        /// Supports different blending modes for various face regions and quality requirements.
        /// </summary>
        /// <param name="originalImage">The original background image to blend the face into</param>
        /// <param name="faceTexture">The generated face texture to blend onto the original image</param>
        /// <param name="faceBbox">The face bounding box coordinates (x1, y1, x2, y2) in original image space</param>
        /// <param name="cropBox">The expanded crop box coordinates for face region processing</param>
        /// <param name="precomputedBlurredMask">The precomputed blurred segmentation mask for smooth blending</param>
        /// <param name="precomputedFaceLarge">The precomputed expanded face region for efficient processing</param>
        /// <param name="extraMargin">Additional margin to apply for specific versions (e.g., v15 bottom margin)</param>
        /// <param name="mode">The blending mode: "jaw" for talking head with mouth region, "raw" for basic face blending</param>
        /// <returns>A seamlessly blended frame with the generated face composited into the original image</returns>
        /// <exception cref="System.Exception">Thrown when precomputed blurred mask is null or invalid</exception>
        public static Frame BlendFaceWithOriginal(
            Frame originalImage, 
            Frame faceTexture,  
            Vector4 faceBbox, 
            Vector4 cropBox,
            Frame precomputedBlurredMask, 
            Frame precomputedFaceLarge, 
            float extraMargin,
            string mode = "raw")
        {
            if (precomputedBlurredMask.data == null)
            {
                throw new System.Exception("Precomputed blurred mask is null - segmentation data must be generated during avatar processing");
            }

            // Apply version-specific adjustments to face bounding box
            Vector4 adjustedFaceBbox = faceBbox;
            if (mode == "jaw") // v15 mode with extra margin for mouth area
            {
                adjustedFaceBbox.w = Mathf.Min(adjustedFaceBbox.w + extraMargin, originalImage.height);
            }
            
            // Use optimal path with precomputed masks for maximum performance
            var result = ApplySegmentationMaskWithPrecomputedMasks(
                originalImage, 
                faceTexture, 
                adjustedFaceBbox, 
                cropBox,
                precomputedBlurredMask, 
                precomputedFaceLarge, 
                mode);
            return result;
        }

        #endregion

        #region Public Methods - Mask Processing

        /// <summary>
        /// Creates a small mask by cropping the BiSeNet segmentation result to the face region.
        /// This method extracts the relevant portion of the full segmentation mask that corresponds
        /// to the detected face area.
        /// </summary>
        /// <param name="maskData">The full BiSeNet segmentation mask to crop from</param>
        /// <param name="faceBbox">The face bounding box coordinates (x1, y1, x2, y2)</param>
        /// <param name="cropBox">The expanded crop box coordinates used for face region processing</param>
        /// <returns>A cropped mask frame containing only the face region segmentation</returns>
        public static Frame CreateSmallMask(Frame maskData, Vector4 faceBbox, Rect cropBox)
        {
            float x = faceBbox.x;
            float y = faceBbox.y;
            float x1 = faceBbox.z;
            float y1 = faceBbox.w;
            float x_s = cropBox.x;
            float y_s = cropBox.y;
            
            // Calculate crop region in BiSeNet mask coordinate space
            Rect cropRect = new(
                x - x_s,
                y - y_s,
                x1 - x,  // width = x1 - x
                y1 - y   // height = y1 - y
            );
            
            return FrameUtils.CropFrame(maskData, cropRect);
        }

        /// <summary>
        /// Creates a full-size mask by pasting the small mask into a blank canvas of original dimensions.
        /// This method reconstructs the segmentation mask at the original image resolution by placing
        /// the cropped face mask back into its proper position.
        /// </summary>
        /// <param name="originalMaskData">The original mask data for dimension reference</param>
        /// <param name="smallMaskData">The small cropped mask to paste into the full canvas</param>
        /// <param name="faceBbox">The face bounding box coordinates for positioning</param>
        /// <param name="cropBox">The crop box coordinates for offset calculation</param>
        /// <returns>A full-size mask frame with the small mask pasted at the correct position</returns>
        public static unsafe Frame CreateFullMask(
            Frame originalMaskData, 
            Frame smallMaskData, 
            Vector4 faceBbox, 
            Rect cropBox)
        {
            int width = originalMaskData.width;
            int height = originalMaskData.height;
            int rowSizeBytes = width * 3; // RGB24: 3 bytes per pixel
            
            // Create blank mask canvas with RGB24 format for efficiency
            var fullMaskData = new Frame(new byte[width * height * 3], width, height);
            float x = faceBbox.x;
            float y = faceBbox.y;
            float x_s = cropBox.x;
            float y_s = cropBox.y;
            
            int pasteX = Mathf.RoundToInt(x - x_s);
            int pasteY = Mathf.RoundToInt(y - y_s);
            
            // Calculate valid paste region to avoid bounds checking in inner loops
            int startX = Mathf.Max(0, pasteX);
            int endX = Mathf.Min(width, pasteX + smallMaskData.width);
            int startY = Mathf.Max(0, pasteY);
            int endY = Mathf.Min(height, pasteY + smallMaskData.height);
            
            fixed (byte* fullMaskPtr = fullMaskData.data)
            fixed (byte* smallMaskPtr = smallMaskData.data)
            {
                byte* fullMaskPtrLocal = fullMaskPtr;
                byte* smallMaskPtrLocal = smallMaskPtr;
                
                // Initialize canvas to black using parallel memory clearing
                System.Threading.Tasks.Parallel.For(0, height, y =>
                {
                    byte* targetRowPtr = fullMaskPtrLocal + y * rowSizeBytes;
                    UnsafeUtility.MemClear(targetRowPtr, rowSizeBytes);
                });
                
                // Parallel paste operation for optimal performance
                System.Threading.Tasks.Parallel.For(startY, endY, targetY =>
                {
                    int sourceY = targetY - pasteY;
                    if (sourceY >= 0 && sourceY < smallMaskData.height)
                    {
                        byte* targetRowPtr = fullMaskPtrLocal + targetY * rowSizeBytes;
                        byte* sourceRowPtr = smallMaskPtrLocal + sourceY * smallMaskData.width * 3;
                        
                        // Optimize for full row copies when possible
                        if (pasteX == 0 && startX == 0 && endX == width && smallMaskData.width == width)
                        {
                            // Bulk copy entire row using native memory copy (fastest path)
                            UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, smallMaskData.width * 3);
                        }
                        else
                        {
                            // Partial row copy with pixel-by-pixel processing
                            for (int targetX = startX; targetX < endX; targetX++)
                            {
                                int sourceX = targetX - pasteX;
                                if (sourceX >= 0 && sourceX < smallMaskData.width)
                                {
                                    byte* targetPixelPtr = targetRowPtr + targetX * 3;
                                    byte* sourcePixelPtr = sourceRowPtr + sourceX * 3;
                                    
                                    // Use red channel as grayscale value for all RGB components
                                    byte grayValue = sourcePixelPtr[0]; // Red channel
                                    targetPixelPtr[0] = grayValue; // R
                                    targetPixelPtr[1] = grayValue; // G
                                    targetPixelPtr[2] = grayValue; // B
                                }
                            }
                        }
                    }
                });
            }
            
            return fullMaskData;
        }

        /// <summary>
        /// Applies upper boundary ratio to preserve the upper portion of the face from blending.
        /// This method creates a mask that protects the upper face region (eyes, nose, forehead) while
        /// allowing blending in the lower region (mouth, chin) for natural talking head animation.
        /// Uses optimized parallel processing with unsafe memory operations for maximum performance.
        /// </summary>
        /// <param name="maskData">The input mask to apply boundary ratio processing to</param>
        /// <param name="upperBoundaryRatio">The ratio of the upper face to preserve (0.5 = top 50%)</param>
        /// <returns>A processed mask with upper boundary ratio applied for selective face blending</returns>
        public static unsafe Frame ApplyUpperBoundaryRatio(Frame maskData, float upperBoundaryRatio)
        {
            // Create result with RGB24 format for efficiency
            var resultData = new Frame(new byte[maskData.width * maskData.height * 3], maskData.width, maskData.height);
            
            // Calculate boundary line using standard image coordinates
            // Y=0 is top, Y=height-1 is bottom
            // upperBoundaryRatio=0.5 means preserve top 50% of face (upper half)
            int boundaryY = Mathf.RoundToInt(maskData.height * upperBoundaryRatio);
            int rowSizeBytes = maskData.width * 3; // RGB24: 3 bytes per pixel
            
            // Use unsafe pointers for maximum performance with parallel processing
            fixed (byte* maskPtr = maskData.data)
            fixed (byte* resultPtr = resultData.data)
            {
                byte* maskPtrLocal = maskPtr;
                byte* resultPtrLocal = resultPtr;
                
                // Parallel processing of rows for optimal performance
                System.Threading.Tasks.Parallel.For(0, maskData.height, y =>
                {
                    byte* sourceRowPtr = maskPtrLocal + y * rowSizeBytes;
                    byte* targetRowPtr = resultPtrLocal + y * rowSizeBytes;
                    
                    // Process based on vertical position in image
                    if (y < boundaryY) // Upper part of face (preserve original - eyes, nose, forehead)
                    {
                        // Zero out entire row to preserve original face features
                        UnsafeUtility.MemClear(targetRowPtr, rowSizeBytes);
                    }
                    else // Lower part of face (talking area - mouth, chin)
                    {
                        // Copy mask data to allow blending in mouth/chin area
                        UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, rowSizeBytes);
                    }
                });
            }
            
            return resultData;
        }
        
        /// <summary>
        /// Applies Gaussian blur to a segmentation mask for smooth edge transitions.
        /// This method softens mask edges to eliminate harsh boundaries and artifacts during face blending,
        /// calculating the optimal kernel size based on mask dimensions for natural-looking results.
        /// </summary>
        /// <param name="mask">The input segmentation mask to blur</param>
        /// <returns>A Gaussian-blurred mask with smooth edges optimized for seamless face blending</returns>
        public static Frame ApplyGaussianBlurToMask(Frame mask)
        {
            // Calculate adaptive blur kernel size based on mask dimensions
            const float blurFactor = 0.08f; // Standard blur factor
            int kernelSize = Mathf.RoundToInt(blurFactor * mask.width / 2) * 2 + 1; // Ensure odd kernel size
            kernelSize = Mathf.Max(kernelSize, 15); // Minimum kernel size for effective smoothing

            var blurredMaskData = FrameUtils.ApplySimpleGaussianBlur(mask, kernelSize);
            return blurredMaskData;
        }

        #endregion

        #region Private Methods - Image Composition

        /// <summary>
        /// Applies segmentation mask to blend face with original image using precomputed optimization data.
        /// This method implements the complete face blending pipeline using cached segmentation masks
        /// for maximum performance, avoiding expensive mask regeneration for each frame.
        /// </summary>
        /// <param name="originalImage">The original background image</param>
        /// <param name="faceTexture">The generated face texture to blend</param>
        /// <param name="faceBbox">The adjusted face bounding box coordinates</param>
        /// <param name="cropBox">The crop box coordinates for face region processing</param>
        /// <param name="precomputedBlurredMask">The precomputed blurred mask for smooth blending</param>
        /// <param name="precomputedFaceLarge">The precomputed expanded face region</param>
        /// <param name="mode">The blending mode for different face region processing</param>
        /// <returns>The final blended frame with seamless face composition</returns>
        private static Frame ApplySegmentationMaskWithPrecomputedMasks(
            Frame originalImage, 
            Frame faceTexture,  
            Vector4 faceBbox, 
            Vector4 cropBox,
            Frame precomputedBlurredMask, 
            Frame precomputedFaceLarge, 
            string mode = "raw")
        {
            // Step 1: Resize face texture to match face bounding box dimensions
            int faceWidth = (int)(faceBbox.z - faceBbox.x);
            int faceHeight = (int)(faceBbox.w - faceBbox.y);
            var resizedFace = FrameUtils.ResizeFrame(faceTexture, faceWidth, faceHeight);

            // Step 2: Paste the resized face into precomputed face_large at relative position
            var cropRect = new Rect(cropBox.x, cropBox.y, cropBox.z - cropBox.x, cropBox.w - cropBox.y);
            var faceLargeWithFace = PasteFaceIntoLarge(
                precomputedFaceLarge, 
                resizedFace, 
                faceBbox, 
                cropRect);

            // Step 3: Composite final result using precomputed blurred mask for alpha blending
            var result = CompositeWithMask(
                originalImage, 
                faceLargeWithFace, 
                faceBbox, 
                precomputedBlurredMask, 
                cropBox);

            return result;
        }

        /// <summary>
        /// Pastes generated face into the expanded face region at the correct relative position.
        /// This method handles precise positioning and efficient memory operations for face placement
        /// within the larger face context, using parallel processing for optimal performance.
        /// </summary>
        /// <param name="faceLarge">The expanded face region to paste into</param>
        /// <param name="generatedFace">The generated face to paste</param>
        /// <param name="faceBbox">The face bounding box for position calculation</param>
        /// <param name="cropBox">The crop box rectangle for offset calculation</param>
        /// <returns>The face region with the generated face pasted at the correct position</returns>
        private static unsafe Frame PasteFaceIntoLarge(
            Frame faceLarge, 
            Frame generatedFace, 
            Vector4 faceBbox,
            Rect cropBox)
        {
            // Calculate relative position within face_large using exact integer calculation
            int relativeX = (int)(faceBbox.x - cropBox.x);
            int relativeY = (int)(faceBbox.y - cropBox.y);
            
            // Create result with RGB24 format for efficiency
            var result = new Frame(new byte[faceLarge.width * faceLarge.height * 3], faceLarge.width, faceLarge.height);
            
            // Get unsafe pointers for direct memory operations
            fixed (byte* faceLargePtr = faceLarge.data)
            fixed (byte* generatedFacePtr = generatedFace.data)
            fixed (byte* resultPtr = result.data)
            {
                int resultWidth = faceLarge.width;
                int resultHeight = faceLarge.height;
                int faceWidth = generatedFace.width;
                int faceHeight = generatedFace.height;

                byte* faceLargePtrLocal = faceLargePtr;
                byte* generatedFacePtrLocal = generatedFacePtr;
                byte* resultPtrLocal = resultPtr;

                // First, copy the entire faceLarge to result using parallel memory copy
                System.Threading.Tasks.Parallel.For(0, resultHeight, y =>
                {
                    byte* sourceRowPtr = faceLargePtrLocal + y * resultWidth * 3;
                    byte* targetRowPtr = resultPtrLocal + y * resultWidth * 3;
                    UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, resultWidth * 3);
                });

                // Calculate valid paste region to avoid bounds checking in inner loops
                int startX = Mathf.Max(0, relativeX);
                int endX = Mathf.Min(resultWidth, relativeX + faceWidth);
                int startY = Mathf.Max(0, relativeY);
                int endY = Mathf.Min(resultHeight, relativeY + faceHeight);
                
                // Parallel paste operation for optimal performance
                System.Threading.Tasks.Parallel.For(startY, endY, targetY =>
                {
                    int sourceY = targetY - relativeY;
                    if (sourceY >= 0 && sourceY < faceHeight)
                    {
                        byte* targetRowPtr = resultPtrLocal + targetY * resultWidth * 3;
                        byte* sourceRowPtr = generatedFacePtrLocal + sourceY * faceWidth * 3;
                        
                        // Optimize for full row copies when dimensions align perfectly
                        if (relativeX == 0 && startX == 0 && endX == resultWidth && faceWidth == resultWidth)
                        {
                            // Bulk copy entire row using native memory copy (fastest path)
                            UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, faceWidth * 3);
                        }
                        else
                        {
                            // Partial row copy using optimized chunk copying
                            int copyStartX = startX;
                            int copyEndX = endX;
                            int sourceStartX = copyStartX - relativeX;
                            int copyWidth = copyEndX - copyStartX;
                            
                            if (copyWidth > 0 && sourceStartX >= 0)
                            {
                                byte* targetPtr = targetRowPtr + copyStartX * 3;
                                byte* sourcePtr = sourceRowPtr + sourceStartX * 3;
                                
                                // Use memory copy for contiguous pixel chunks (much faster than pixel-by-pixel)
                                UnsafeUtility.MemCpy(targetPtr, sourcePtr, copyWidth * 3);
                            }
                        }
                    }
                });
            }
            return result;
        }
        
        /// <summary>
        /// Composites images using alpha blending with the blurred segmentation mask.
        /// This method implements alpha blending functionality with optimized unsafe memory operations
        /// and parallel processing for real-time performance during face composition.
        /// </summary>
        /// <param name="originalImage">The original background image</param>
        /// <param name="faceLarge">The face region to blend onto the background</param>
        /// <param name="faceBbox">The face bounding box for positioning reference</param>
        /// <param name="blurredMask">The blurred segmentation mask for alpha blending</param>
        /// <param name="cropBox">The crop box coordinates for positioning</param>
        /// <returns>The final composited image with seamless alpha-blended face integration</returns>
        private static unsafe Frame CompositeWithMask(
            Frame originalImage, 
            Frame faceLarge, 
            Vector4 faceBbox,
            Frame blurredMask, 
            Vector4 cropBox)
        {
            int resultWidth = originalImage.width;
            int resultHeight = originalImage.height;
            int blendWidth = faceLarge.width;
            int blendHeight = faceLarge.height;
            
            // Create result with RGB24 format for efficiency
            var result = new Frame(new byte[resultWidth * resultHeight * 3], resultWidth, resultHeight);
            
            // Get unsafe pointers for direct memory operations
            fixed (byte* originalPtr = originalImage.data)
            fixed (byte* faceLargePtr = faceLarge.data)
            fixed (byte* maskPtr = blurredMask.data)
            fixed (byte* resultPtr = result.data)
            {
                byte* originalPtrLocal = originalPtr;
                byte* faceLargePtrLocal = faceLargePtr;
                byte* maskPtrLocal = maskPtr;
                byte* resultPtrLocal = resultPtr;

                // Calculate paste position
                int pasteX = (int)cropBox.x;
                int pasteY = (int)cropBox.y;
                
                // First, copy the entire original image to result using parallel memory copy
                System.Threading.Tasks.Parallel.For(0, resultHeight, y =>
                {
                    byte* sourceRowPtr = originalPtrLocal + y * resultWidth * 3;
                    byte* targetRowPtr = resultPtrLocal + y * resultWidth * 3;
                    UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, resultWidth * 3);
                });
                
                // Calculate valid blend region to avoid bounds checking in inner loops
                int startX = Mathf.Max(0, pasteX);
                int endX = Mathf.Min(resultWidth, pasteX + blendWidth);
                int startY = Mathf.Max(0, pasteY);
                int endY = Mathf.Min(resultHeight, pasteY + blendHeight);
                
                // Parallel alpha blending with optimized performance
                System.Threading.Tasks.Parallel.For(startY, endY, targetY =>
                {
                    int sourceY = targetY - pasteY;
                    if (sourceY >= 0 && sourceY < blendHeight)
                    {
                        byte* resultRowPtr = resultPtrLocal + targetY * resultWidth * 3;
                        byte* faceLargeRowPtr = faceLargePtrLocal + sourceY * blendWidth * 3;
                        byte* maskRowPtr = maskPtrLocal + sourceY * blendWidth * 3;
                        
                        // Process pixels in this row with optimized alpha blending
                        for (int targetX = startX; targetX < endX; targetX++)
                        {
                            int sourceX = targetX - pasteX;
                            if (sourceX >= 0 && sourceX < blendWidth)
                            {
                                byte* targetPixel = resultRowPtr + targetX * 3;
                                byte* sourcePixel = faceLargeRowPtr + sourceX * 3;
                                byte* maskPixel = maskRowPtr + sourceX * 3;
                                
                                // Get mask alpha (use red channel as alpha, convert to 0-1 range)
                                float alpha = maskPixel[0] / 255.0f;
                                
                                if (alpha > 0.001f) // Small threshold to avoid unnecessary blending operations
                                {
                                    // Optimized alpha blend: result = foreground * alpha + background * (1 - alpha)
                                    float invAlpha = 1.0f - alpha;
                                    
                                    // Blend RGB channels with efficient byte arithmetic
                                    targetPixel[0] = (byte)(sourcePixel[0] * alpha + targetPixel[0] * invAlpha); // R
                                    targetPixel[1] = (byte)(sourcePixel[1] * alpha + targetPixel[1] * invAlpha); // G
                                    targetPixel[2] = (byte)(sourcePixel[2] * alpha + targetPixel[2] * invAlpha); // B
                                }
                            }
                        }
                    }
                });
            }
            
            return result;
        }

        #endregion
    }
}
