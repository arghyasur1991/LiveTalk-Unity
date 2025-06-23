using System.Collections;
using UnityEngine;

namespace LiveTalk.API
{
    using System;
    using UnityEngine.Video;
    using Utils;
    
    public class LiveTalkController : MonoBehaviour
    {
        private DrivingFramesStream _drivingFramesStream;
        private bool _frameProcessed = false;
        private long _currentFrameIndex = 0;
        private long _totalFramesToProcess = 0;

        public DrivingFramesStream DrivingFramesStream => _drivingFramesStream;

        public void LoadDrivingFrames(string[] frameFiles)
        {
            _drivingFramesStream = new DrivingFramesStream(frameFiles.Length);
            _drivingFramesStream.TotalExpectedFrames = frameFiles.Length;
            StartCoroutine(LoadDrivingFramesAsync(frameFiles, _drivingFramesStream));
        }

        public void LoadDrivingFrames(VideoPlayer videoPlayer)
        {
            if (videoPlayer == null || videoPlayer.clip == null)
            {
                Debug.LogError("[LiveTalkController] VideoPlayer or clip is null");
                return;
            }

            if (videoPlayer.clip.frameCount == 0)
            {
                Debug.LogError("[LiveTalkController] VideoPlayer clip has no frames");
                return;
            }

            _drivingFramesStream = new DrivingFramesStream((int)videoPlayer.clip.frameCount);
            _drivingFramesStream.TotalExpectedFrames = (int)videoPlayer.clip.frameCount;
            StartCoroutine(LoadDrivingFramesAsync(videoPlayer, _drivingFramesStream));
        }

        private void OnFrameReady(VideoPlayer source, long frameIndex)
        {
            try
            {
                Debug.Log($"[LiveTalkController] Frame ready: {frameIndex}");
                
                // Read the render texture
                RenderTexture.active = source.targetTexture;
                Texture2D frameTexture = new Texture2D(source.targetTexture.width, source.targetTexture.height, TextureFormat.RGB24, false);
                frameTexture.ReadPixels(new Rect(0, 0, source.targetTexture.width, source.targetTexture.height), 0, 0);
                frameTexture.Apply();
                RenderTexture.active = null;
                
                // Convert to RGB24 format if needed
                var rgbTexture = TextureUtils.ConvertTexture2DToRGB24(frameTexture);
                rgbTexture.name = $"VideoFrame_{frameIndex:D6}";
                
                // Enqueue the frame
                _drivingFramesStream.loadQueue.Enqueue(rgbTexture);
                
                // Clean up original texture if different
                if (rgbTexture != frameTexture)
                {
                    UnityEngine.Object.DestroyImmediate(frameTexture);
                }
                
                Debug.Log($"[LiveTalkController] Processed frame {frameIndex}, queue count: {_drivingFramesStream.QueueCount}");
                
                // Check if we've processed all frames
                if (frameIndex >= _totalFramesToProcess - 1)
                {
                    // Stop the video and disable frameReady events
                    source.Stop();
                    source.sendFrameReadyEvents = false;
                    source.frameReady -= OnFrameReady;
                    
                    // Cleanup render texture
                    if (source.targetTexture != null)
                    {
                        var renderTexture = source.targetTexture;
                        source.targetTexture = null;
                        renderTexture.Release();
                        UnityEngine.Object.DestroyImmediate(renderTexture);
                    }
                    
                    _drivingFramesStream.LoadingFinished = true;
                    Debug.Log($"[LiveTalkController] Finished loading driving frames from video, {_drivingFramesStream.QueueCount} frames queued");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[LiveTalkController] Error processing frame {frameIndex}: {e.Message}");
            }
        }

        private IEnumerator LoadDrivingFramesAsync(VideoPlayer videoPlayer, DrivingFramesStream stream)
        {
            if (videoPlayer == null || videoPlayer.clip == null)
            {
                Debug.LogError("[LiveTalkController] VideoPlayer or clip is null");
                stream.LoadingFinished = true;
                yield break;
            }

            // Setup VideoPlayer for frame extraction
            RenderTexture renderTexture = new RenderTexture((int)videoPlayer.clip.width, (int)videoPlayer.clip.height, 0);
            videoPlayer.targetTexture = renderTexture;
            
            // Prepare the video player
            videoPlayer.isLooping = false;
            videoPlayer.playOnAwake = false;
            videoPlayer.skipOnDrop = false;
            
            // Prepare and wait for video to be ready
            videoPlayer.Prepare();
            yield return new WaitUntil(() => videoPlayer.isPrepared);

            // Initialize frame processing variables
            _totalFramesToProcess = (long)videoPlayer.clip.frameCount;
            
            // Enable frameReady events and subscribe to the event - following Unity docs pattern
            videoPlayer.sendFrameReadyEvents = true;
            videoPlayer.frameReady += OnFrameReady;
            
            // Start playback - this will trigger frameReady events for each frame
            videoPlayer.Play();
            
            // Wait for all frames to be processed
            yield return new WaitUntil(() => stream.LoadingFinished);
        }

        /// <summary>
        /// Load driving frames asynchronously and add them to the stream
        /// </summary>
        private IEnumerator LoadDrivingFramesAsync(string[] frameFiles, DrivingFramesStream stream)
        {
            for (int i = 0; i < frameFiles.Length; i++)
            {
                string filePath = frameFiles[i];
                
                // Load frame data asynchronously - outside try-catch to avoid yield in try-catch
                var loadFileTask = System.IO.File.ReadAllBytesAsync(filePath);
                yield return new WaitUntil(() => loadFileTask.IsCompleted);
                
                try
                {
                    if (loadFileTask.IsFaulted)
                    {
                        Debug.LogError($"[LivePortraitMuseTalkAPI] Error loading driving frame {filePath}: {loadFileTask.Exception?.GetBaseException().Message}");
                        continue;
                    }
                    
                    byte[] fileData = loadFileTask.Result;
                    Texture2D texture = new(2, 2);
                    
                    if (texture.LoadImage(fileData))
                    {
                        texture.name = System.IO.Path.GetFileNameWithoutExtension(filePath);
                        var rgbTexture = TextureUtils.ConvertTexture2DToRGB24(texture);
                        stream.loadQueue.Enqueue(rgbTexture);
                        
                        // Clean up original texture if different
                        if (rgbTexture != texture)
                        {
                            UnityEngine.Object.DestroyImmediate(texture);
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"[LivePortraitMuseTalkAPI] Failed to load image: {filePath}");
                        UnityEngine.Object.DestroyImmediate(texture);
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"[LivePortraitMuseTalkAPI] Error processing driving frame {filePath}: {e.Message}");
                }
                yield return null;
            }

            stream.LoadingFinished = true;
            Debug.Log($"[LivePortraitMuseTalkAPI] Finished loading driving frames, {stream.QueueCount} frames queued");
        }
    }
}