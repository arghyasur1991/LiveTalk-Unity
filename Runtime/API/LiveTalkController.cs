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
        private long _totalFramesToProcess = 0;

        public DrivingFramesStream DrivingFramesStream => _drivingFramesStream;

        public void LoadDrivingFrames(string[] frameFiles)
        {
            _drivingFramesStream = new DrivingFramesStream(frameFiles.Length);
            _drivingFramesStream.TotalExpectedFrames = frameFiles.Length;
            StartCoroutine(LoadDrivingFramesAsync(frameFiles, _drivingFramesStream));
        }

        public void LoadDrivingFrames(VideoPlayer videoPlayer, int maxFrames = -1)
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

            var frameCount = Mathf.Min(maxFrames, (int)videoPlayer.clip.frameCount);
            _drivingFramesStream = new DrivingFramesStream(frameCount)
            {
                TotalExpectedFrames = frameCount
            };
            StartCoroutine(LoadDrivingFramesAsync(videoPlayer, _drivingFramesStream, maxFrames));
        }

        private void OnFrameReady(VideoPlayer source, long frameIndex)
        {
            try
            {
                Debug.Log($"[LiveTalkController] Frame ready: {frameIndex}");
                
                // Check if the VideoPlayer texture is available
                if (source.texture == null)
                {
                    Debug.LogWarning($"[LiveTalkController] VideoPlayer.texture is null for frame {frameIndex}");
                    return;
                }
                
                // Use VideoPlayer.texture directly instead of custom RenderTexture
                RenderTexture videoTexture = source.texture as RenderTexture;
                if (videoTexture == null)
                {
                    Debug.LogWarning($"[LiveTalkController] VideoPlayer.texture is not a RenderTexture for frame {frameIndex}");
                    return;
                }
                
                Debug.Log($"[LiveTalkController] Video texture size: {videoTexture.width}x{videoTexture.height}");
                
                // Read from the VideoPlayer's texture
                RenderTexture.active = videoTexture;
                Texture2D frameTexture = new Texture2D(videoTexture.width, videoTexture.height, TextureFormat.RGB24, false);
                frameTexture.ReadPixels(new Rect(0, 0, videoTexture.width, videoTexture.height), 0, 0);
                frameTexture.Apply();
                RenderTexture.active = null;
                
                // Convert to RGB24 format if needed
                var rgbTexture = TextureUtils.ConvertTexture2DToRGB24(frameTexture);
                rgbTexture.name = $"VideoFrame_{frameIndex:D6}";
                
                // Enqueue the frame
                if (_drivingFramesStream.QueueCount < _drivingFramesStream.TotalExpectedFrames)
                {
                    _drivingFramesStream.loadQueue.Enqueue(rgbTexture);
                }
                
                // Clean up original texture if different
                if (rgbTexture != frameTexture)
                {
                    DestroyImmediate(frameTexture);
                }
                
                Debug.Log($"[LiveTalkController] Processed frame {frameIndex}, queue count: {_drivingFramesStream.QueueCount}");
                
                // Check if we've processed all frames
                if (frameIndex >= _totalFramesToProcess - 1)
                {
                    // Stop the video and disable frameReady events
                    source.Stop();
                    source.sendFrameReadyEvents = false;
                    source.frameReady -= OnFrameReady;
                    
                    _drivingFramesStream.LoadingFinished = true;
                    Debug.Log($"[LiveTalkController] Finished loading driving frames from video, {_drivingFramesStream.QueueCount} frames queued");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[LiveTalkController] Error processing frame {frameIndex}: {e.Message}");
            }
        }

        private IEnumerator LoadDrivingFramesAsync(VideoPlayer videoPlayer, DrivingFramesStream stream, int maxFrames = -1)
        {
            if (videoPlayer == null || videoPlayer.clip == null)
            {
                Debug.LogError("[LiveTalkController] VideoPlayer or clip is null");
                stream.LoadingFinished = true;
                yield break;
            }

            // Prepare the video player - don't set custom targetTexture, let VideoPlayer handle it
            videoPlayer.isLooping = false;
            videoPlayer.playOnAwake = false;
            videoPlayer.skipOnDrop = false;
            
            // Prepare and wait for video to be ready
            videoPlayer.Prepare();
            yield return new WaitUntil(() => videoPlayer.isPrepared);
            
            // Wait an additional frame to ensure everything is set up
            yield return null;

            // Initialize frame processing variables
            _totalFramesToProcess = Mathf.Min(maxFrames, (int)videoPlayer.clip.frameCount);
            
            Debug.Log($"[LiveTalkController] Video prepared. Frame count: {_totalFramesToProcess}, Video size: {videoPlayer.clip.width}x{videoPlayer.clip.height}");
            
            // Enable frameReady events and subscribe to the event
            videoPlayer.sendFrameReadyEvents = true;
            videoPlayer.frameReady += OnFrameReady;
            
            // Start playback - this will trigger frameReady events for each frame
            videoPlayer.Play();
            
            // Wait a frame after starting playback
            yield return null;
            
            Debug.Log($"[LiveTalkController] VideoPlayer started. IsPlaying: {videoPlayer.isPlaying}");
            
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