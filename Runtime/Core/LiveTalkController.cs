using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Video;

namespace LiveTalk.Core
{
    using API;
    using Utils;
    
    internal class LiveTalkController : MonoBehaviour
    {
        private FrameStream _drivingFramesStream;
        private long _totalFramesToProcess = 0;

        public FrameStream DrivingFramesStream => _drivingFramesStream;

        public void LoadDrivingFrames(string[] frameFiles)
        {
            _drivingFramesStream = new FrameStream(frameFiles.Length)
            {
                TotalExpectedFrames = frameFiles.Length
            };
            StartCoroutine(Produce(LoadDrivingFramesAsync(frameFiles, _drivingFramesStream), _drivingFramesStream,
                "LiveTalkController.LoadDrivingFrames(files)"));
        }

        /// <summary>
        /// Runs a frame producer under <see cref="TaskYield.Guard"/> so a fault
        /// is logged, recorded on the stream, and can never leave a consumer
        /// waiting on an unfinished stream.
        /// </summary>
        internal static IEnumerator Produce(IEnumerator producer, FrameStream stream, string context)
        {
            return TaskYield.Guard(producer, ex =>
            {
                // The stream is the only channel back to whoever is draining
                // it, and they may not check Error — so log here as well.
                Logger.LogError($"[{context}] Frame producer failed: {ex.GetType().Name}: {ex.Message}");
                stream.Fail(ex);
            }, context);
        }

        public void LoadDrivingFrames(VideoPlayer videoPlayer, int maxFrames = -1)
        {
            if (videoPlayer == null || videoPlayer.clip == null)
            {
                Logger.LogError("[LiveTalkController] VideoPlayer or clip is null");
                return;
            }

            if (videoPlayer.clip.frameCount == 0)
            {
                Logger.LogError("[LiveTalkController] VideoPlayer clip has no frames");
                return;
            }
            var frameCount = maxFrames == -1 ? (int)videoPlayer.clip.frameCount :
                                     Mathf.Min(maxFrames, (int)videoPlayer.clip.frameCount);
            _drivingFramesStream = new FrameStream(frameCount)
            {
                TotalExpectedFrames = frameCount
            };
            StartCoroutine(Produce(LoadDrivingFramesAsync(videoPlayer, _drivingFramesStream, maxFrames), _drivingFramesStream,
                "LiveTalkController.LoadDrivingFrames(video)"));
        }

        private void OnFrameReady(VideoPlayer source, long frameIndex)
        {
            try
            {
                Logger.LogVerbose($"[LiveTalkController] Frame ready: {frameIndex}");
                
                // Check if the VideoPlayer texture is available
                if (source.texture == null)
                {
                    Logger.LogWarning($"[LiveTalkController] VideoPlayer.texture is null for frame {frameIndex}");
                    return;
                }
                
                // Use VideoPlayer.texture directly instead of custom RenderTexture
                RenderTexture videoTexture = source.texture as RenderTexture;
                if (videoTexture == null)
                {
                    Logger.LogWarning($"[LiveTalkController] VideoPlayer.texture is not a RenderTexture for frame {frameIndex}");
                    return;
                }
                
                Logger.LogVerbose($"[LiveTalkController] Video texture size: {videoTexture.width}x{videoTexture.height}");
                
                // Read from the VideoPlayer's texture
                RenderTexture.active = videoTexture;
                Texture2D frameTexture = new(videoTexture.width, videoTexture.height, TextureFormat.RGB24, false);
                frameTexture.ReadPixels(new Rect(0, 0, videoTexture.width, videoTexture.height), 0, 0);
                frameTexture.Apply();
                RenderTexture.active = null;
                
                // Convert to RGB24 format if needed
                var rgbTexture = TextureUtils.ConvertTexture2DToRGB24(frameTexture);
                rgbTexture.name = $"VideoFrame_{frameIndex:D6}";
                
                // Enqueue the frame
                if (frameIndex < _drivingFramesStream.TotalExpectedFrames)
                {
                    _drivingFramesStream.Queue.Enqueue(rgbTexture);
                }
                
                // Clean up original texture if different
                if (rgbTexture != frameTexture)
                {
                    DestroyImmediate(frameTexture);
                }
                
                Logger.LogVerbose($"[LiveTalkController] Processed frame {frameIndex}");
                
                // Check if we've processed all frames
                if (frameIndex >= _totalFramesToProcess - 1)
                {
                    // Stop the video and disable frameReady events
                    source.Stop();
                    source.sendFrameReadyEvents = false;
                    source.frameReady -= OnFrameReady;
                    
                    _drivingFramesStream.Finished = true;
                    Logger.LogVerbose($"[LiveTalkController] Finished loading driving frames from video");
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[LiveTalkController] Error processing frame {frameIndex}: {e.Message}");
            }
        }

        private IEnumerator LoadDrivingFramesAsync(VideoPlayer videoPlayer, FrameStream stream, int maxFrames = -1)
        {
            if (videoPlayer == null || videoPlayer.clip == null)
            {
                Logger.LogError("[LiveTalkController] VideoPlayer or clip is null");
                stream.Finished = true;
                yield break;
            }

            try
            {
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
                _totalFramesToProcess = maxFrames == -1 ? (int)videoPlayer.clip.frameCount :
                                         Mathf.Min(maxFrames, (int)videoPlayer.clip.frameCount);

                Logger.LogVerbose($"[LiveTalkController] Video prepared. Frame count: {_totalFramesToProcess}, Video size: {videoPlayer.clip.width}x{videoPlayer.clip.height}");

                // Enable frameReady events and subscribe to the event
                videoPlayer.sendFrameReadyEvents = true;
                videoPlayer.frameReady += OnFrameReady;

                // Start playback - this will trigger frameReady events for each frame
                videoPlayer.Play();

                // Wait a frame after starting playback
                yield return null;

                Logger.LogVerbose($"[LiveTalkController] VideoPlayer started. IsPlaying: {videoPlayer.isPlaying}");

                // Wait for all frames to be processed
                yield return new WaitUntil(() => stream.Finished);
            }
            finally
            {
                // Normally OnFrameReady has already stopped the player and
                // unsubscribed; this covers the coroutine being stopped or
                // disposed mid-way, so a dead loader never leaves the event
                // hooked or the consumer waiting.
                videoPlayer.frameReady -= OnFrameReady;
                videoPlayer.sendFrameReadyEvents = false;
                stream.Finished = true;
            }
        }

        /// <summary>
        /// Load driving frames asynchronously and add them to the stream
        /// </summary>
        private IEnumerator LoadDrivingFramesAsync(string[] frameFiles, FrameStream stream)
        {
            try
            {
                for (int i = 0; i < frameFiles.Length; i++)
                {
                    string filePath = frameFiles[i];

                    // An unreadable driving frame fails the load. Skipping it
                    // used to produce a silently shorter animation; the bridge
                    // logs the file and rethrows so the consumer sees an error.
                    byte[] fileData = null;
                    yield return TaskYield.Wait(File.ReadAllBytesAsync(filePath), b => fileData = b,
                        $"LiveTalkController.LoadDrivingFrame {filePath}");

                    try
                    {
                        Texture2D texture = new(2, 2);

                        if (texture.LoadImage(fileData))
                        {
                            texture.name = Path.GetFileNameWithoutExtension(filePath);
                            var rgbTexture = TextureUtils.ConvertTexture2DToRGB24(texture);
                            stream.Queue.Enqueue(rgbTexture);

                            // Clean up original texture if different
                            if (rgbTexture != texture)
                            {
                                UnityEngine.Object.DestroyImmediate(texture);
                            }
                        }
                        else
                        {
                            Logger.LogWarning($"[LivePortraitMuseTalkAPI] Failed to load image: {filePath}");
                            UnityEngine.Object.DestroyImmediate(texture);
                        }
                    }
                    catch (Exception e)
                    {
                        Logger.LogError($"[LivePortraitMuseTalkAPI] Error processing driving frame {filePath}: {e.Message}");
                    }
                    yield return null;
                }

                Logger.Log($"[LivePortraitMuseTalkAPI] Finished loading driving frames");
            }
            finally
            {
                stream.Finished = true;
            }
        }
    }
}
