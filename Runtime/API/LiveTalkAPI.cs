using SparkTTS;
using SparkTTS.Utils;
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.Video;

namespace LiveTalk.API
{
    using Core;
    using Utils;

    #region Public Data Types

    public enum LogLevel
    {
        VERBOSE,
        INFO,
        WARNING,
        ERROR,
    }

    public enum CreationMode
    {
        /// <summary>
        /// Only generate voice.
        /// </summary>
        VoiceOnly,
        /// <summary>
        /// Generate voice and single expression.
        /// </summary>
        SingleExpression,
        /// <summary>
        /// Generate voice and all expressions.
        /// </summary>
        AllExpressions,
    }

    public enum MemoryUsage
    {
        /// <summary>
        /// Not Recommended (Not enough extra quality trade-off with performance and memory usage)
        /// Requires all FP32 models to be packaged in StreamingAssets manually.
        /// </summary>
        Quality, 
        /// <summary>
        /// For desktop devices. Slower first use of any model.
        /// </summary>
        Performance, 
        /// <summary>
        /// Recommended for desktop devices. Prevent unnecessary model loading at startup. Default.
        /// </summary>
        Balanced, 
        /// <summary>
        /// For mobile devices (Not recommended for desktop)
        /// </summary>
        Optimal, 
    }

    /// <summary>
    /// Class for all stream operations in the LiveTalk pipeline.
    /// Provides common functionality for frame streaming including queuing, cancellation, and synchronization.
    /// </summary>
    public class FrameStream
    {
        #region Properties

        /// <summary>
        /// Gets or sets the total number of frames expected to be processed through this stream.
        /// </summary>
        public int TotalExpectedFrames { get; set; }
        
        /// <summary>
        /// Gets a value indicating whether more frames are available for processing.
        /// This includes both queued frames and frames that are still being loaded.
        /// </summary>
        public bool HasMoreFrames => !Finished || !Queue.IsEmpty;

        /// <summary>
        /// Gets or sets a value indicating whether the stream processing is finished.
        /// </summary>
        internal bool Finished { get; set; } = false;

        /// <summary>
        /// Gets the internal concurrent queue used for frame storage and retrieval.
        /// </summary>
        internal readonly ConcurrentQueue<Texture2D> Queue = new();

        /// <summary>
        /// Gets the cancellation token source for stream operations.
        /// </summary>
        internal CancellationTokenSource CancellationTokenSource { get; } = new();

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the FrameStream class with the specified total expected frames.
        /// </summary>
        /// <param name="totalExpectedFrames">The total number of frames expected to be processed</param>
        public FrameStream(int totalExpectedFrames)
        {
            TotalExpectedFrames = totalExpectedFrames;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Creates a yield instruction that waits until the next frame is available.
        /// The frame is then accessible through the FrameAwaiter.Texture property.
        /// </summary>
        /// <returns>A FrameAwaiter instance that can be used in Unity coroutines</returns>
        public FrameAwaiter WaitForNext() => new(Queue);

        /// <summary>
        /// Attempts to retrieve the next frame from the queue without blocking.
        /// </summary>
        /// <param name="texture">The retrieved texture, or null if no frame is available</param>
        /// <returns>True if a frame was successfully retrieved; false if the queue is empty</returns>
        public bool TryGetNext(out Texture2D texture) => Queue.TryDequeue(out texture);

        #endregion
    }

    /// <summary>
    /// Custom yield instruction for Unity coroutines that waits for and delivers texture frames.
    /// This class provides non-blocking frame retrieval with automatic Unity integration.
    /// </summary>
    public sealed class FrameAwaiter : CustomYieldInstruction
    {
        #region Private Fields

        private readonly ConcurrentQueue<Texture2D> _queue;

        #endregion

        #region Properties

        /// <summary>
        /// Gets the texture that was retrieved from the queue.
        /// This property is set when a frame becomes available.
        /// </summary>
        public Texture2D Texture { get; private set; }

        /// <summary>
        /// Gets a value indicating whether the coroutine should continue waiting.
        /// Returns false when a frame is available, allowing the coroutine to continue.
        /// </summary>
        public override bool keepWaiting
        {
            get
            {
                if (_queue.TryDequeue(out var texture))
                {
                    Texture = texture;
                    return false; // Stop waiting - caller resumes
                }
                return true; // Keep waiting this frame
            }
        }

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the FrameAwaiter class with the specified queue.
        /// </summary>
        /// <param name="queue">The concurrent queue to monitor for available frames</param>
        public FrameAwaiter(ConcurrentQueue<Texture2D> queue)
        {
            _queue = queue;
        }

        #endregion
    }

    #endregion

    #region Main API Classes

    /// <summary>
    /// Integrated API that combines LivePortrait and MuseTalk for comprehensive talking head generation.
    /// Provides a unified interface for avatar animation with motion transfer and audio synchronization.
    /// 
    /// Character Format:
    /// - Bundle Format (.bundle) - macOS only:
    ///   - Character data is stored in a .bundle directory that appears as a single file in macOS Finder
    ///   - Contains Info.plist for proper macOS package metadata
    ///   - Automatically used on macOS platforms
    /// - Folder Format - Universal:
    ///   - Character data is stored in a regular directory
    ///   
    /// Workflow:
    /// 1. LivePortrait: Generate animated textures from source image and driving frames
    /// 2. MuseTalk: Apply lip synchronization to animated textures using audio
    /// 3. Character: Create a character with the specified parameters.
    /// 
    /// This class orchestrates the complete pipeline for realistic talking head video generation.
    /// </summary>
    public class LiveTalkAPI : IDisposable
    {
        #region Private Fields
        private LivePortraitInference _livePortrait = null;
        private MuseTalkInference _museTalk = null;
        private LiveTalkConfig _config;
        private LiveTalkController _controller;
        private GameObject _liveTalkInstance;
        private bool _disposed = false;
        private bool _initialized = false;

        #endregion

        #region Properties

        /// <summary>
        /// Gets the LiveTalk configuration for internal operations.
        /// </summary>
        public string CharacterSaveLocation => Character.saveLocation;

        /// <summary>
        /// Gets the MuseTalk inference engine for internal operations.
        /// </summary>
        internal MuseTalkInference MuseTalk => _museTalk;

        /// <summary>
        /// Gets the LivePortrait inference engine for internal operations.
        /// </summary>
        internal LivePortraitInference LivePortrait => _livePortrait;

        /// <summary>
        /// Gets the GameObject that contains the LiveTalkAPI components.
        /// </summary>
        internal GameObject Object => _liveTalkInstance;

        /// <summary>
        /// Gets the LiveTalk configuration for internal operations.
        /// </summary>
        internal LiveTalkConfig Config => _config;

        #endregion

        #region Constructor
        public static LiveTalkAPI Instance { get; private set; } = new();

        /// <summary>
        /// Initializes a new instance of the LiveTalkAPI class with the specified configuration and controller.
        /// </summary>
        /// <param name="logLevel">The logging level for the API (defaults to WARNING)</param>
        /// <param name="characterSaveLocation">The location to save the generated characters</param>
        /// <param name="parentModelPath">The parent path for model files (defaults to StreamingAssets if empty)</param>
        /// <param name="memoryUsage">The memory usage level for the API (defaults to Balanced)</param>
        public void Initialize(
            LogLevel logLevel = LogLevel.INFO,
            string characterSaveLocation = "",
            string parentModelPath = "",
            MemoryUsage memoryUsage = MemoryUsage.Balanced)
        {
            if (_initialized)
            {
                Logger.LogWarning("[LiveTalkAPI] Already initialized");
                return;
            }

            if (string.IsNullOrEmpty(parentModelPath))
            {
                parentModelPath = Application.streamingAssetsPath;
            }

            if (string.IsNullOrEmpty(characterSaveLocation))
            {
                characterSaveLocation = Path.Combine(Application.persistentDataPath, "Characters");
            }
            
            _config = new LiveTalkConfig(parentModelPath, logLevel, memoryUsage);
            Logger.LogLevel = _config.LogLevel;
            _livePortrait = new LivePortraitInference(_config);
            _museTalk = new MuseTalkInference(_config);
            ModelUtils.Initialize(_config.LogLevel);

            Character.saveLocation = characterSaveLocation;
            _liveTalkInstance = new GameObject("LiveTalkAPI");
            _controller = _liveTalkInstance.AddComponent<LiveTalkController>();
            _liveTalkInstance.AddComponent<VideoPlayer>();

            var sparkTTSLogLevel = _config.LogLevel switch
            {
                LogLevel.VERBOSE => SparkTTS.Utils.LogLevel.VERBOSE,
                LogLevel.INFO => SparkTTS.Utils.LogLevel.INFO,
                LogLevel.WARNING => SparkTTS.Utils.LogLevel.WARNING,
                LogLevel.ERROR => SparkTTS.Utils.LogLevel.ERROR,
                _ => SparkTTS.Utils.LogLevel.WARNING,
            };
            bool optimalMemoryUsage = memoryUsage == MemoryUsage.Optimal;
            CharacterVoiceFactory.Initialize(sparkTTSLogLevel, optimalMemoryUsage);
            _initialized = true;
        }

        #endregion

        #region Public Methods - LivePortrait Animation

        /// <summary>
        /// Generates animated textures from a source image and a list of driving frames.
        /// This method processes all driving frames synchronously and provides streaming output.
        /// </summary>
        /// <param name="sourceImage">The source image containing the face to animate</param>
        /// <param name="drivingFrames">The list of driving frames that define the motion to transfer</param>
        /// <returns>An FrameStream for receiving generated animated frames</returns>
        /// <exception cref="ArgumentException">Thrown when source image or driving frames are null</exception>
        public FrameStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, List<Texture2D> drivingFrames)
        {
            ValidateAnimationInputs(sourceImage, drivingFrames);
            Logger.Log($"[LiveTalkAPI] Generating animated textures: {drivingFrames.Count} driving frames");
            
            var outputStream = new FrameStream(drivingFrames.Count);
            var inputStream = CreateInputStreamFromFrames(drivingFrames);
            inputStream.TotalExpectedFrames = drivingFrames.Count;
            
            _controller.StartCoroutine(_livePortrait.GenerateAsync(sourceImage, outputStream, inputStream));
            return outputStream;
        }

        /// <summary>
        /// Generates animated textures from a source image and a video player's frames.
        /// This method provides pipelined processing for efficient video-based animation.
        /// </summary>
        /// <param name="sourceImage">The source image containing the face to animate</param>
        /// <param name="videoPlayer">The video player containing the driving frames</param>
        /// <param name="maxFrames">The maximum number of frames to process (-1 for all frames)</param>
        /// <returns>An FrameStream for receiving generated animated frames</returns>
        /// <exception cref="ArgumentException">Thrown when source image or video player is null</exception>
        public FrameStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, VideoPlayer videoPlayer, int maxFrames = -1)
        {
            ValidateAnimationInputs(sourceImage, videoPlayer);

            int frameCount = CalculateFrameCount(videoPlayer, maxFrames);
            Logger.Log($"[LiveTalkAPI] Generating animated textures: {frameCount} driving frames from video");
            
            var outputStream = new FrameStream(frameCount);
            _controller.LoadDrivingFrames(videoPlayer, maxFrames);
            _controller.StartCoroutine(_livePortrait.GenerateAsync(sourceImage, outputStream, _controller.DrivingFramesStream));
            
            return outputStream;
        }

        /// <summary>
        /// Generates animated textures from a source image and driving frames loaded from a directory path.
        /// This method provides efficient file-based animation processing with streaming output.
        /// </summary>
        /// <param name="sourceImage">The source image containing the face to animate</param>
        /// <param name="drivingFramesPath">The path to the directory containing driving frame images</param>
        /// <param name="maxFrames">The maximum number of frames to process (-1 for all frames)</param>
        /// <returns>An FrameStream for receiving generated animated frames</returns>
        /// <exception cref="ArgumentException">Thrown when source image or path is invalid, or no frames are found</exception>
        public FrameStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, string drivingFramesPath, int maxFrames = -1)
        {
            ValidateAnimationInputs(sourceImage, drivingFramesPath);

            var frameFiles = GetFrameFiles(drivingFramesPath, maxFrames);
            Logger.Log($"[LiveTalkAPI] Generating animated textures: {frameFiles.Length} driving frames from directory");
            
            var outputStream = new FrameStream(frameFiles.Length);
            _controller.LoadDrivingFrames(frameFiles);
            _controller.StartCoroutine(_livePortrait.GenerateAsync(sourceImage, outputStream, _controller.DrivingFramesStream));
            
            return outputStream;
        }

        #endregion

        #region Public Methods - MuseTalk Lip Synchronization

        /// <summary>
        /// Generates talking head video with lip synchronization using avatar frames and audio.
        /// This method combines facial animation with audio-driven lip movements for realistic speech synthesis.
        /// </summary>
        /// <param name="avatarTexture">The primary avatar texture to animate</param>
        /// <param name="talkingHeadFolderPath">The path to additional avatar frames for variation</param>
        /// <param name="audioClip">The audio clip to synchronize with the generated video</param>
        /// <returns>An FrameStream for receiving generated talking head frames</returns>
        /// <exception cref="ArgumentException">Thrown when avatar texture or audio clip is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when the controller is not available</exception>
        public FrameStream GenerateTalkingHeadAsync(Texture2D avatarTexture, string talkingHeadFolderPath, AudioClip audioClip)
        {
            ValidateControllerAvailability();
            ValidateTalkingHeadInputs(avatarTexture, audioClip);
            
            Logger.Log($"[LiveTalkAPI] Generating talking head: {audioClip.name} ({audioClip.length:F2}s)");
            
            var avatarTextures = LoadAvatarTextures(avatarTexture, talkingHeadFolderPath);
            int estimatedFrames = EstimateFrameCount(audioClip);
            
            var outputStream = new FrameStream(estimatedFrames);
            _controller.StartCoroutine(_museTalk.GenerateAsync(avatarTextures, audioClip, outputStream));
            
            return outputStream;
        }

        /// <summary>
        /// Generates talking head video using preloaded avatar data for optimized performance.
        /// This method bypasses avatar processing and directly generates frames from precomputed data.
        /// </summary>
        /// <param name="avatarData">The preloaded avatar data containing face regions and latent representations</param>
        /// <param name="audioClip">The audio clip to synchronize with the generated video</param>
        /// <returns>An FrameStream for receiving generated talking head frames</returns>
        /// <exception cref="ArgumentException">Thrown when audio clip is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when the controller is not available</exception>
        internal FrameStream GenerateTalkingHeadWithPreloadedData(AvatarData avatarData, AudioClip audioClip)
        {
            ValidateControllerAvailability();
            ValidateTalkingHeadInputs(null, audioClip);
            
            Logger.Log($"[LiveTalkAPI] Generating talking head: {audioClip.name} ({audioClip.length:F2}s)");
            
            int estimatedFrames = EstimateFrameCount(audioClip);
            var outputStream = new FrameStream(estimatedFrames);
            
            _controller.StartCoroutine(_museTalk.GenerateWithPreloadedDataAsync(audioClip, avatarData, outputStream));
            return outputStream;
        }

        #endregion

        #region Public Methods - Character Creation

        /// <summary>
        /// Creates a new character with the specified parameters
        /// </summary>
        /// <param name="name">The name of the character</param>
        /// <param name="gender">The gender of the character</param>
        /// <param name="image">The image of the character</param>
        /// <param name="pitch">The pitch of the character</param>
        /// <param name="speed">The speed of the character</param>
        /// <param name="intro">The intro of the character</param>
        /// <param name="voicePromptPath">The path to the voice prompt</param>
        /// <param name="onComplete">Callback when character is successfully created</param>
        /// <param name="onError">Callback when an error occurs</param>
        public IEnumerator CreateCharacterAsync(
            string name,
            Gender gender,
            Texture2D image,
            Pitch pitch,
            Speed speed,
            string intro,
            string voicePromptPath,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            return CreateCharacterAsync(name, gender, image, pitch, speed, intro, voicePromptPath, onComplete, onError, CreationMode.AllExpressions);
        }

        /// <summary>
        /// Creates a new character with the specified parameters
        /// </summary>
        /// <param name="name">The name of the character</param>
        /// <param name="gender">The gender of the character</param>
        /// <param name="image">The image of the character</param>
        /// <param name="pitch">The pitch of the character</param>
        /// <param name="speed">The speed of the character</param>
        /// <param name="intro">The intro of the character</param>
        /// <param name="voicePromptPath">The path to the voice prompt</param>
        /// <param name="onComplete">Callback when character is successfully created</param>
        /// <param name="onError">Callback when an error occurs</param>
        /// <param name="creationMode">The creation mode to use</param>
        /// <param name="useBundle">Whether to use a bundle if possible</param>
        public IEnumerator CreateCharacterAsync(
            string name,
            Gender gender,
            Texture2D image,
            Pitch pitch,
            Speed speed,
            string intro,
            string voicePromptPath,
            Action<Character> onComplete,
            Action<Exception> onError,
            CreationMode creationMode,
            bool useBundle = true)
        {
            if (!_initialized)
            {
                onError?.Invoke(new Exception("CharacterFactory not initialized. Call Initialize() first."));
                yield break;
            }

            var character = new Character(name, gender, image, pitch, speed, intro);
            useBundle = useBundle && CanUseBundle();
            yield return character.CreateAvatarAsync(voicePromptPath, useBundle, creationMode);
            onComplete?.Invoke(character);
        }

        /// <summary>
        /// Load a character from the saveLocation using the character GUID
        /// </summary>
        /// <param name="characterId">The GUID/hash of the character to load</param>
        /// <param name="onComplete">Callback when character is successfully loaded</param>
        /// <param name="onError">Callback when an error occurs</param>
        public IEnumerator LoadCharacterAsync(
            string characterId,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            if (!_initialized)
            {
                onError?.Invoke(new Exception("CharacterFactory not initialized. Call Initialize() first."));
                yield break;
            }

            if (string.IsNullOrEmpty(characterId))
            {
                onError?.Invoke(new ArgumentException("Character ID cannot be null or empty."));
                yield break;
            }

            yield return Character.LoadCharacterAsync(characterId, onComplete, onError);
        }

        /// <summary>
        /// Get all available character IDs from the saveLocation
        /// </summary>
        /// <returns>Array of character GUIDs/hashes</returns>
        public string[] GetAvailableCharacterIds()
        {
            if (!_initialized || string.IsNullOrEmpty(Character.saveLocation))
            {
                return new string[0];
            }

            try
            {
                if (!Directory.Exists(Character.saveLocation))
                {
                    return new string[0];
                }

                var directories = Directory.GetDirectories(Character.saveLocation);
                var characterIds = new List<string>();

                foreach (var dir in directories)
                {
                    string dirName = Path.GetFileName(dir);
                    string characterConfigPath = Path.Combine(dir, "character.json");
                    
                    // Only include directories that have a character.json file
                    if (File.Exists(characterConfigPath))
                    {
                        // Remove .bundle extension if present to get the actual character ID
                        if (dirName.EndsWith(".bundle"))
                        {
                            dirName = dirName.Substring(0, dirName.Length - 7); // Remove ".bundle"
                        }
                        
                        // Avoid duplicates (in case both folder and bundle exist for same character)
                        if (!characterIds.Contains(dirName))
                        {
                            characterIds.Add(dirName);
                        }
                    }
                }

                return characterIds.ToArray();
            }
            catch (Exception ex)
            {
                Logger.LogError($"[CharacterFactory] Error getting available character IDs: {ex.Message}");
                return new string[0];
            }
        }

        /// <summary>
        /// Get the recommended character format for the current platform
        /// </summary>
        /// <returns>True if bundle format is recommended, false for folder format</returns>
        public static bool CanUseBundle()
        {
            return Application.platform == RuntimePlatform.OSXEditor || Application.platform == RuntimePlatform.OSXPlayer;
        }

        #endregion

        #region Private Methods - Input Validation

        /// <summary>
        /// Validates common animation inputs for source image and driving frames.
        /// </summary>
        /// <param name="sourceImage">The source image to validate</param>
        /// <param name="drivingFrames">The driving frames to validate</param>
        /// <exception cref="ArgumentException">Thrown when inputs are invalid</exception>
        private static void ValidateAnimationInputs(Texture2D sourceImage, List<Texture2D> drivingFrames)
        {
            if (sourceImage == null || drivingFrames == null)
                throw new ArgumentException("Invalid input: source image and driving frames are required");
        }

        /// <summary>
        /// Validates animation inputs for source image and video player.
        /// </summary>
        /// <param name="sourceImage">The source image to validate</param>
        /// <param name="videoPlayer">The video player to validate</param>
        /// <exception cref="ArgumentException">Thrown when inputs are invalid</exception>
        private static void ValidateAnimationInputs(Texture2D sourceImage, VideoPlayer videoPlayer)
        {
            if (sourceImage == null || videoPlayer == null)
                throw new ArgumentException("Invalid input: source image and video player are required");
        }

        /// <summary>
        /// Validates animation inputs for source image and driving frames path.
        /// </summary>
        /// <param name="sourceImage">The source image to validate</param>
        /// <param name="drivingFramesPath">The driving frames path to validate</param>
        /// <exception cref="ArgumentException">Thrown when inputs are invalid</exception>
        private static void ValidateAnimationInputs(Texture2D sourceImage, string drivingFramesPath)
        {
            if (sourceImage == null || string.IsNullOrEmpty(drivingFramesPath))
                throw new ArgumentException("Invalid input: source image and driving frames path are required");
        }

        /// <summary>
        /// Validates talking head inputs for avatar texture and audio clip.
        /// </summary>
        /// <param name="avatarTexture">The avatar texture to validate (can be null for preloaded data)</param>
        /// <param name="audioClip">The audio clip to validate</param>
        /// <exception cref="ArgumentException">Thrown when audio clip is null</exception>
        private static void ValidateTalkingHeadInputs(Texture2D avatarTexture, AudioClip audioClip)
        {
            if (audioClip == null)
                throw new ArgumentException("Audio clip is required");
        }

        /// <summary>
        /// Validates that the controller is available for streaming operations.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown when the controller is not available</exception>
        private void ValidateControllerAvailability()
        {
            if (_controller == null)
                throw new InvalidOperationException("Controller is required for streaming operations. Use constructor with LiveTalkController parameter.");
        }

        #endregion

        #region Private Methods - Helper Functions

        /// <summary>
        /// Creates an input stream from a list of driving frames.
        /// </summary>
        /// <param name="drivingFrames">The driving frames to populate the stream with</param>
        /// <returns>An InputStream populated with the driving frames</returns>
        private static FrameStream CreateInputStreamFromFrames(List<Texture2D> drivingFrames)
        {
            var inputStream = new FrameStream(drivingFrames.Count);
            foreach (var frame in drivingFrames)
            {
                inputStream.Queue.Enqueue(frame);
            }
            return inputStream;
        }

        /// <summary>
        /// Calculates the frame count based on video player settings and maximum frame limit.
        /// </summary>
        /// <param name="videoPlayer">The video player to get frame count from</param>
        /// <param name="maxFrames">The maximum number of frames to process</param>
        /// <returns>The calculated frame count</returns>
        private static int CalculateFrameCount(VideoPlayer videoPlayer, int maxFrames)
        {
            return maxFrames == -1 ? (int)videoPlayer.clip.frameCount : 
                   Mathf.Min(maxFrames, (int)videoPlayer.clip.frameCount);
        }

        /// <summary>
        /// Gets frame files from the specified directory path with optional frame limit.
        /// </summary>
        /// <param name="drivingFramesPath">The path to search for frame files</param>
        /// <param name="maxFrames">The maximum number of frames to retrieve</param>
        /// <returns>An array of frame file paths</returns>
        /// <exception cref="ArgumentException">Thrown when no frames are found</exception>
        private static string[] GetFrameFiles(string drivingFramesPath, int maxFrames)
        {
            var frameFiles = FileUtils.GetFrameFiles(drivingFramesPath, maxFrames);
            if (frameFiles.Length == 0)
            {
                throw new ArgumentException($"No driving frames found in path: {drivingFramesPath}");
            }
            return frameFiles;
        }

        /// <summary>
        /// Loads avatar textures from a primary texture and optional folder path.
        /// </summary>
        /// <param name="avatarTexture">The primary avatar texture</param>
        /// <param name="talkingHeadFolderPath">The optional folder path for additional textures</param>
        /// <returns>A list of avatar textures for processing</returns>
        private static List<Texture2D> LoadAvatarTextures(Texture2D avatarTexture, string talkingHeadFolderPath)
        {
            var avatarTextures = FileUtils.LoadFramesFromFolder(talkingHeadFolderPath);
            if (avatarTextures == null || avatarTextures.Count == 0)
            {
                avatarTexture = TextureUtils.ConvertTexture2DToRGB24(avatarTexture);
                avatarTextures = new List<Texture2D> { avatarTexture };
            }
            return avatarTextures;
        }

        /// <summary>
        /// Estimates the number of frames needed based on audio clip duration.
        /// </summary>
        /// <param name="audioClip">The audio clip to estimate frame count for</param>
        /// <returns>The estimated frame count based on 25 FPS</returns>
        private static int EstimateFrameCount(AudioClip audioClip)
        {
            return Mathf.CeilToInt(audioClip.length * 25f); // ~25 FPS estimate
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Releases all resources used by the LiveTalkAPI instance.
        /// Disposes of all inference engines and cleans up model utilities.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases the unmanaged resources used by the LiveTalkAPI and optionally releases the managed resources.
        /// </summary>
        /// <param name="disposing">True to release both managed and unmanaged resources; false to release only unmanaged resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    _livePortrait?.Dispose();
                    _museTalk?.Dispose();
                    ModelUtils.Dispose();
                }
                
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for the LiveTalkAPI class.
        /// </summary>
        ~LiveTalkAPI()
        {
            Dispose(false);
        }

        #endregion
    }

    #endregion
}
