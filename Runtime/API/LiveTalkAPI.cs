using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace LiveTalk.API
{
    using System.IO;
    using Core;
    using UnityEngine.Video;
    using Utils;

    public sealed class OutputStream
    {
        public int TotalExpectedFrames { get; set; }

        public OutputStream(int totalExpectedFrames)
        {
            TotalExpectedFrames = totalExpectedFrames;
        }

        internal readonly ConcurrentQueue<Texture2D> queue = new();
        internal CancellationTokenSource cts = new();

        public bool Finished { get; internal set; }

        /// Non-blocking poll. Returns false if no frame is ready yet.
        public bool TryGetNext(out Texture2D tex) => queue.TryDequeue(out tex);

        /// Yield instruction that waits until the *next* frame exists,
        /// then exposes it through the .Texture property.
        public FrameAwaiter WaitForNext() => new(queue);
    }

    /// Custom yield instruction that delivers one Texture2D.
    public sealed class FrameAwaiter : CustomYieldInstruction
    {
        private readonly ConcurrentQueue<Texture2D> _q;
        public Texture2D Texture { get; private set; }

        public FrameAwaiter(ConcurrentQueue<Texture2D> q) => _q = q;

        public override bool keepWaiting
        {
            get
            {
                if (_q.TryDequeue(out var tex))
                {
                    Texture = tex;
                    return false;          // stop waiting – caller resumes
                }
                return true;               // keep waiting this frame
            }
        }
    }

    /// <summary>
    /// Stream for driving frames input - similar to output stream but for input processing
    /// </summary>
    public sealed class DrivingFramesStream
    {
        public int TotalExpectedFrames { get; set; }
        public bool LoadingFinished { get; internal set; }
        public bool ProcessingFinished { get; internal set; }

        public DrivingFramesStream(int totalExpectedFrames)
        {
            TotalExpectedFrames = totalExpectedFrames;
        }

        internal readonly ConcurrentQueue<Texture2D> loadQueue = new();
        internal CancellationTokenSource cts = new();

        /// Non-blocking poll. Returns false if no frame is ready yet.
        public bool TryGetNext(out Texture2D tex) => loadQueue.TryDequeue(out tex);

        /// Check if frames are available for processing
        public bool HasFramesAvailable => !loadQueue.IsEmpty;

        /// Get current queue count
        public int QueueCount => loadQueue.Count;

        /// Yield instruction that waits until the *next* frame exists,
        /// then exposes it through the .Texture property.
        public FrameAwaiter WaitForNext() => new(loadQueue);

        /// Check if we have more frames to process
        public bool HasMoreFrames => !LoadingFinished || HasFramesAvailable;
    }

    /// <summary>
    /// Integrated API that combines LivePortrait and MuseTalk for complete talking head generation
    /// 
    /// Workflow:
    /// 1. LivePortrait: Generate animated textures from single source image + driving frames
    /// 2. MuseTalk: Apply lip sync to the animated textures using audio
    /// 
    /// This matches the user's requested workflow exactly
    /// </summary>
    public class LiveTalkAPI : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        private readonly LivePortraitInference _livePortrait;
        private readonly MuseTalkInference _museTalk;
        private readonly LiveTalkConfig _config;
        private bool _disposed = false;
        private readonly LiveTalkController _avatarController;
        
        /// <summary>
        /// Initialize the integrated API with configuration
        /// </summary>
        public LiveTalkAPI(LiveTalkConfig config, LiveTalkController avatarController)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _avatarController = avatarController ?? throw new ArgumentNullException(nameof(avatarController));
            
            _livePortrait = new LivePortraitInference(_config);
            _museTalk = new MuseTalkInference(_config);
        }
        
        /// <summary>
        /// Generate animated textures only using LivePortrait (SYNCHRONOUS) - List<Texture2D> overload
        /// </summary>
        public OutputStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, List<Texture2D> drivingFrames)
        {                
            if (sourceImage == null || drivingFrames == null)
                throw new ArgumentException("Invalid input: source image and driving frames are required");
                
            Logger.Log($"[LivePortraitMuseTalkAPI] Generating animated textures (SYNC): {drivingFrames.Count} driving frames");
            
            var stream = new OutputStream(drivingFrames.Count);
            var drivingStream = new DrivingFramesStream(drivingFrames.Count);
            foreach (var frame in drivingFrames)
            {
                drivingStream.loadQueue.Enqueue(frame);
            }
            _avatarController.StartCoroutine(_livePortrait.GenerateAsync(sourceImage, stream, drivingStream));
            return stream;
        }

        public OutputStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, VideoPlayer videoPlayer, int maxFrames = -1)
        {                
            if (sourceImage == null || videoPlayer == null)
                throw new ArgumentException("Invalid input: source image and video player are required");

            Logger.Log($"[LivePortraitMuseTalkAPI] Starting pipelined processing: {videoPlayer.clip.frameCount} driving frames");
            var frameCount = Mathf.Min(maxFrames, (int)videoPlayer.clip.frameCount);
            var stream = new OutputStream(frameCount);
            _avatarController.LoadDrivingFrames(videoPlayer, maxFrames);
            _avatarController.StartCoroutine(
                _livePortrait.GenerateAsync(sourceImage, stream, _avatarController.DrivingFramesStream));
            return stream;
        }

        public OutputStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, string drivingFramesPath, int maxFrames = -1)
        {
            if (sourceImage == null || string.IsNullOrEmpty(drivingFramesPath))
                throw new ArgumentException("Invalid input: source image and driving frames path are required");

            // Get frame count first to estimate total frames
            var frameFiles = FileUtils.GetFrameFiles(drivingFramesPath, maxFrames); // Send maxFrames > 0 to load some frames
            if (frameFiles.Length == 0)
            {
                throw new ArgumentException($"No driving frames found in path: {drivingFramesPath}");
            }

            Logger.Log($"[LivePortraitMuseTalkAPI] Starting pipelined processing: {frameFiles.Length} driving frames");
            
            var stream = new OutputStream(frameFiles.Length);
            _avatarController.LoadDrivingFrames(frameFiles);
            _avatarController.StartCoroutine(
                _livePortrait.GenerateAsync(sourceImage, stream, _avatarController.DrivingFramesStream));
            return stream;
        }

        /// <summary>
        /// Generate talking head video with streaming output (NEW STREAMING API)
        /// Similar to LivePortrait's streaming approach - yields frames as they're generated
        /// </summary>
        /// <param name="avatarTexture">Avatar image texture</param>
        /// <param name="audioClip">Speech audio clip</param>
        /// <param name="batchSize">Processing batch size (default: 4)</param>
        /// <returns>MuseTalkStream for receiving frames as they're generated</returns>
        public OutputStream GenerateTalkingHeadAsync(Texture2D avatarTexture, string talkingHeadFolderPath, AudioClip audioClip)
        {
            if (_avatarController == null)
                throw new InvalidOperationException("Avatar controller is required for streaming operations. Use constructor with AvatarController parameter.");
                
            if (avatarTexture == null || audioClip == null)
                throw new ArgumentException("Avatar texture and audio clip are required");
                
            Logger.Log($"[MuseTalkFactory] Starting streaming generation: {audioClip.name} ({audioClip.length:F2}s)");
            List<Texture2D> avatarTextures = FileUtils.LoadFramesFromFolder(talkingHeadFolderPath);
            if (avatarTextures == null || avatarTextures.Count == 0)
            {
                avatarTexture = TextureUtils.ConvertTexture2DToRGB24(avatarTexture);
                avatarTextures = new List<Texture2D> { avatarTexture };
            }

            var input = new MuseTalkInput(avatarTextures.ToArray(), audioClip)
            {
                BatchSize = _config.BatchSize
            };
            
            // Estimate frame count based on audio length (approximation)
            int estimatedFrames = Mathf.CeilToInt(audioClip.length * 25f); // ~25 FPS estimate
            var stream = new OutputStream(estimatedFrames);
            
            _avatarController.StartCoroutine(_museTalk.GenerateAsync(input, stream));
            return stream;
        }



        
        public void Dispose()
        {
            if (!_disposed)
            {
                _livePortrait?.Dispose();
                _museTalk?.Dispose();
                _disposed = true;
                Logger.Log("[LivePortraitMuseTalkAPI] Disposed");
            }
        }
        
        ~LiveTalkAPI()
        {
            Dispose();
        }
    }
    
    /// <summary>
    /// Factory for creating LivePortraitMuseTalkAPI instances
    /// </summary>
    public static class LivePortraitMuseTalkFactory
    {
        /// <summary>
        /// Create an instance of the integrated API with default configuration
        /// </summary>
        public static LiveTalkAPI Create(LiveTalkController avatarController, string parentModelPath = "")
        {
            if (string.IsNullOrEmpty(parentModelPath))
            {
                parentModelPath = Application.streamingAssetsPath;
            }
            
            var config = new LiveTalkConfig(parentModelPath);
            return new LiveTalkAPI(config, avatarController);
        }
    }
}
