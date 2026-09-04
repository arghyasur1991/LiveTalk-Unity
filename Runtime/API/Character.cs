using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Video;
using QwenTTS;
using QwenTTS.Audio;
using Newtonsoft.Json;

namespace LiveTalk.API
{
    using Core;
    using Utils;
    public enum Gender
    {
        Male,
        Female
    }

    public enum Pitch
    {
        VeryLow,
        Low,
        Moderate,
        High,
        VeryHigh
    }

    public enum Speed
    {
        VeryLow,
        Low,
        Moderate,
        High,
        VeryHigh
    }

    internal class ProcessFramesResult
    {
        public List<Texture2D> GeneratedFrames { get; set; } = new List<Texture2D>();
        public List<string> GeneratedFramePaths { get; set; } = new List<string>();
    }


    /// <summary>
    /// Configuration data structure for loading characters
    /// </summary>
    [Serializable]
    internal class CharacterConfig
    {
        public string name;
        public Gender gender;
        public Pitch pitch;
        public Speed speed;
        public string intro;
        public string voiceInstruct;
    }

    /// <summary>
    /// Data structures for face data JSON deserialization
    /// </summary>
    [Serializable]
    internal class FaceDataContainer
    {
        public FaceRegionData[] faceRegions;
    }

    [Serializable]
    internal class FaceRegionData
    {
        public bool hasFace;
        public BoundingBoxData boundingBox;
        public BoundingBoxData adjustedFaceBbox;
        public BoundingBoxData cropBox;
        public TextureFilesData textureFiles;
        public TextureDimensionsData textureDimensions;
    }

    [Serializable]
    internal class BoundingBoxData
    {
        public float x;
        public float y;
        public float width;
        public float height;
        public float z; // For Vector4 data
        public float w; // For Vector4 data
    }

    [Serializable]
    internal class TextureFilesData
    {
        public string croppedFace;
        public string faceLarge;
        public string segmentationMask;
        public string original;
        public string maskSmall;
        public string fullMask;
        public string boundaryMask;
        public string blurredMask;
    }

    [Serializable]
    internal class TextureDimensionsData
    {
        public TextureDimension croppedFace;
        public TextureDimension faceLarge;
        public TextureDimension segmentationMask;
        public TextureDimension original;
        public TextureDimension maskSmall;
        public TextureDimension fullMask;
        public TextureDimension boundaryMask;
        public TextureDimension blurredMask;
    }

    [Serializable]
    internal class TextureDimension
    {
        public int width;
        public int height;
    }

    /// <summary>
    /// Data for a specific expression including frames, latents, and face data
    /// </summary>
    internal class ExpressionData
    {
        public AvatarData Data { get; set; } = new AvatarData();
        public string ExpressionName { get; set; }
    }

    /// <summary>
    /// Character class supporting both folder and macOS bundle formats.
    /// 
    /// Bundle Format (.bundle) - macOS only:
    /// - Character data is stored in a .bundle directory that appears as a single file in macOS Finder
    /// - Contains Info.plist for proper macOS package metadata
    /// - Automatically used on macOS platforms
    /// 
    /// Folder Format - Universal:
    /// - Character data is stored in a regular directory
    /// - Works on all platforms (Windows, macOS, Linux)
    /// - Used on non-macOS platforms or when explicitly requested
    /// 
    /// Usage Examples:
    /// // Automatic format selection (bundle on macOS, folder on other platforms)
    /// yield return character.CreateAvatarAsync();
    /// 
    /// // Explicit format selection
    /// yield return character.CreateAvatarAsync(useBundle: true);  // Force bundle format
    /// yield return character.CreateAvatarAsync(useBundle: false); // Force folder format
    /// 
    /// // Check character format
    /// string format = LiveTalkAPI.Instance.GetCharacterFormat(characterId); // Returns "bundle", "folder", or null
    /// bool isBundle = LiveTalkAPI.Instance.IsCharacterBundle(characterId);
    /// bool isFolder = LiveTalkAPI.Instance.IsCharacterFolder(characterId);
    /// </summary>
    public class Character
    {
        public string Name { get; internal set; }
        public Gender Gender { get; internal set; }
        public string CharacterId { get; set; }
        public Texture2D Image { get; internal set; }
        public Pitch Pitch { get; internal set; }
        public Speed Speed { get; internal set; }
        public string VoiceInstruct { get; internal set; }
        /// <summary>
        /// Transcript of the clone reference wav (official ICL <c>ref_text</c>).
        /// Empty means x-vector-only clone.
        /// </summary>
        public string VoiceCloneRefText { get; internal set; }
        /// <summary>
        /// Sample rate for generated speech, or 0 for the TTS model's native rate.
        /// Defaults to 16 kHz because that is what the lip-sync stack consumes.
        /// Voice-only characters have no lip-sync consumer and are set to native,
        /// which matters when the clip is going to be a clone reference: the
        /// speaker encoder reads mel up to 12 kHz, so a 16 kHz round trip throws
        /// away the top of the band the speaker is identified by.
        /// </summary>
        public int SpeechSampleRate { get; set; } = 16000;
        public string Intro { get; internal set; } = "Hello, this is a test message";
        /// <summary>
        /// The rendered take that represents this voice, if one was saved.
        /// A designed voice has no inherent audio, so this is whatever sample
        /// was generated when the character was created.
        /// </summary>
        public AudioClip VoicePromptClip => VoiceSampleClip;
        internal static string saveLocation;
        
        // Loaded character data for inference
        public bool IsDataLoaded { get; internal set; } = false;
        internal string CharacterFolder { get; set; }
        internal Dictionary<int, ExpressionData> LoadedExpressions { get; set; } = new Dictionary<int, ExpressionData>();
        internal QwenVoice LoadedVoice { get; set; }

        /// <summary>Rendered take for this voice, when one was saved with it.</summary>
        internal AudioClip VoiceSampleClip { get; set; }
        internal string DrivingFramesFolder { get; set; }
        internal string VoiceFolder { get; set; }
        
        // CharacterPlayer for animation and playback
        private CharacterPlayer _characterPlayer;
        
        /// <summary>
        /// Gets the CharacterPlayer for this character (creates if needed after data is loaded)
        /// </summary>
        public CharacterPlayer CharacterPlayer
        {
            get
            {
                if (_characterPlayer == null && IsDataLoaded)
                {
                    CreateCharacterPlayer();
                }
                return _characterPlayer;
            }
        }
        
        internal Character(
            string name,
            Gender gender,
            Texture2D image,
            Pitch pitch,
            Speed speed,
            string intro)
        {
            Name = name;
            Gender = gender;
            Image = image;
            Pitch = pitch;
            Speed = speed;
            Intro = intro;
        }

        /// <remarks>
        /// Exactly one of <paramref name="onComplete"/> / <paramref name="onError"/>
        /// fires. A character whose voice folder is missing or fails to load,
        /// or whose expression data cannot be read, is reported through
        /// <paramref name="onError"/> rather than handed back half-loaded.
        /// </remarks>
        public static IEnumerator LoadCharacterAsyncFromPath(
            string characterPath,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            return TaskYield.Guard(LoadCharacterFromPathCore(characterPath, onComplete, onError), onError,
                "Character.LoadCharacterAsyncFromPath");
        }

        private static IEnumerator LoadCharacterFromPathCore(
            string characterPath,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            if (string.IsNullOrEmpty(characterPath))
            {
                onError?.Invoke(new ArgumentException("Character path cannot be null or empty."));
                yield break;
            }
            Logger.Log($"[Character] Loading character data from path: {characterPath}");

            Character loadedCharacter = null;
            Exception loadError = null;

            // Load character data in a coroutine
            yield return LoadCharacterDataCoroutine(characterPath, 
                (character) => loadedCharacter = character,
                (error) => loadError = error);

            if (loadError != null)
            {
                onError?.Invoke(loadError);
            }
            else if (loadedCharacter != null)
            {
                onComplete?.Invoke(loadedCharacter);
            }
            else
            {
                onError?.Invoke(new Exception("Failed to load character: Unknown error"));
            }
        }

        public static IEnumerator LoadCharacterAsyncFromId(
            string characterId,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            if (string.IsNullOrEmpty(characterId))
            {
                onError?.Invoke(new ArgumentException("Character ID cannot be null or empty."));
                yield break;
            }

            // Support both folder and .bundle package formats
            string characterPath = GetCharacterPath(characterId);
            if (characterPath == null)
            {
                onError?.Invoke(new DirectoryNotFoundException($"Character not found: {characterId} (checked both folder and .bundle package)"));
                yield break;
            }

            yield return LoadCharacterAsyncFromPath(characterPath, onComplete, onError);
        }

        /// <summary>
        /// Load only character metadata (image + config JSON) without expressions/voice by ID.
        /// This is a lightweight load for thumbnails and lists.
        /// </summary>
        public static IEnumerator LoadCharacterMetadataAsync(
            string characterId,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            if (string.IsNullOrEmpty(characterId))
            {
                onError?.Invoke(new ArgumentException("Character ID cannot be null or empty."));
                yield break;
            }

            string characterPath = GetCharacterPath(characterId);
            if (characterPath == null)
            {
                onError?.Invoke(new DirectoryNotFoundException($"Character not found: {characterId}"));
                yield break;
            }

            yield return LoadCharacterMetadataFromPathAsync(characterPath, onComplete, onError);
        }

        /// <summary>
        /// Load only character metadata (image + config JSON) without expressions/voice from path.
        /// This is a lightweight load for thumbnails and lists.
        /// </summary>
        public static IEnumerator LoadCharacterMetadataFromPathAsync(
            string characterPath,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            return TaskYield.Guard(LoadCharacterMetadataFromPathCore(characterPath, onComplete, onError), onError,
                "Character.LoadCharacterMetadataFromPathAsync");
        }

        private static IEnumerator LoadCharacterMetadataFromPathCore(
            string characterPath,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            if (string.IsNullOrEmpty(characterPath))
            {
                onError?.Invoke(new ArgumentException("Character path cannot be null or empty."));
                yield break;
            }

            Character character = null;
            Exception loadError = null;

            yield return LoadCharacterMetadataCoroutine(characterPath,
                (c) => character = c,
                (e) => loadError = e);

            if (loadError != null)
            {
                onError?.Invoke(loadError);
            }
            else if (character != null)
            {
                onComplete?.Invoke(character);
            }
            else
            {
                onError?.Invoke(new Exception("Failed to load character metadata: Unknown error"));
            }
        }

        /// <summary>
        /// Create avatar with explicit voice prompt path
        /// </summary>
        /// <param name="voicePromptPath">The path to the voice prompt audio file</param>
        /// <param name="useBundle">True to create as macOS bundle, false to create as regular folder</param>
        /// <param name="creationMode">The creation mode to use</param>
        /// <param name="onError">
        /// Receives the failure when avatar generation, voice creation or the
        /// final data load faults. The coroutine then completes normally so the
        /// host's coroutine keeps running; <see cref="IsDataLoaded"/> stays false.
        /// When null the failure is still logged.
        /// </param>
        /// <returns>Coroutine for avatar creation</returns>
        public IEnumerator CreateAvatarAsync(
            string voicePromptPath,
            bool useBundle,
            CreationMode creationMode,
            Action<Exception> onError = null)
        {
            return TaskYield.Guard(CreateAvatarCore(voicePromptPath, useBundle, creationMode), onError,
                "Character.CreateAvatarAsync");
        }

        /// <summary>
        /// Unguarded avatar creation. Faults propagate out of the iterator so
        /// the caller (<see cref="LiveTalkAPI.CreateCharacterAsync"/> or the
        /// public wrapper above) can route them to its own onError and skip
        /// onComplete.
        /// </summary>
        internal IEnumerator CreateAvatarCore(
            string voicePromptPath,
            bool useBundle,
            CreationMode creationMode)
        {
            var start = System.Diagnostics.Stopwatch.StartNew();

            yield return CreateAvatarAsyncInternal(useBundle, creationMode);
            if (!string.IsNullOrEmpty(voicePromptPath))
            {
                yield return TaskYield.Wait(LoadVoiceFromReference(voicePromptPath, VoiceFolder),
                    "Character.LoadVoiceFromReference");
            }
            else
            {
                yield return TaskYield.Wait(GenerateVoiceSample(VoiceFolder),
                    "Character.GenerateVoiceSample");
            }
            yield return LoadData();
            var stop = start.Elapsed;
            Logger.Log($"[Character] Character creation completed for {Name} in {stop.TotalMilliseconds}ms");
        }

        /// <summary>
        /// Generate speech asynchronously using coroutines with optional caching.
        /// Speech audio is automatically cached using the global Cache if enabled.
        /// Provides two callbacks: one when audio is ready, another when animation completes.
        /// Uses queuing to prevent parallel model usage.
        /// </summary>
        /// <param name="text">Text to speak</param>
        /// <param name="expressionIndex">Expression to use, -1 for voice only</param>
        /// <param name="onAudioReady">Callback when audio generation is complete. Called with (FrameStream, AudioClip). 
        /// FrameStream will receive frames as they're generated. Caller can schedule next SpeakAsync here.</param>
        /// <param name="onAnimationComplete">Callback when animation generation is complete. Called with the final FrameStream.
        /// For voice-only mode (expressionIndex=-1), this is called immediately after onAudioReady.</param>
        /// <param name="onError">Callback when an error occurs</param>
        /// <param name="onSpeechChunk">
        /// Optional. Receives speech as it is generated, rather than only when
        /// the line is finished: generation runs slightly faster than playback,
        /// so the first chunk arrives in about a second. Each call carries only
        /// samples not reported before, so appending them in order reproduces
        /// the utterance. Delivered on the main thread.
        ///
        /// Ignored for a cache hit, since there is nothing to stream — the
        /// whole clip is already on disk and arrives via onAudioReady.
        /// </param>
        /// <returns>Coroutine for audio generation</returns>
        /// <remarks>
        /// Every failure reaches <paramref name="onError"/>: a faulted speech
        /// synthesis, a lip-sync model that failed to load, a driving-frame
        /// cache that could not be read. Failures after audio was handed to
        /// <paramref name="onAudioReady"/> still fire <paramref name="onError"/>,
        /// and the <see cref="FrameStream"/> is marked finished with its
        /// <see cref="FrameStream.Error"/> set, so a consumer draining it exits.
        /// <paramref name="onAnimationComplete"/> is not called for a failed
        /// animation.
        /// </remarks>
        public IEnumerator SpeakAsync(
            string text, 
            int expressionIndex = 0,
            Action<FrameStream, AudioClip> onAudioReady = null,
            Action<FrameStream> onAnimationComplete = null,
            Action<Exception> onError = null,
            Action<float[], int> onSpeechChunk = null)
        {
            return TaskYield.Guard(
                SpeakCore(text, expressionIndex, onAudioReady, onAnimationComplete, onError, onSpeechChunk),
                onError,
                "Character.SpeakAsync");
        }

        private IEnumerator SpeakCore(
            string text,
            int expressionIndex,
            Action<FrameStream, AudioClip> onAudioReady,
            Action<FrameStream> onAnimationComplete,
            Action<Exception> onError,
            Action<float[], int> onSpeechChunk)
        {
            var start = System.Diagnostics.Stopwatch.StartNew();
            if (!IsDataLoaded)
            {
                onError?.Invoke(new InvalidOperationException(
                    "Character data not loaded. Use LiveTalkAPI.LoadCharacterAsyncFromId() or CreateCharacterAsync() first, " +
                    "and check that call's onError — a character whose load failed stays unloaded."));
                yield break;
            }

            if (string.IsNullOrEmpty(text))
            {
                onError?.Invoke(new ArgumentException("Text cannot be null or empty."));
                yield break;
            }

            if (expressionIndex != -1 && !LoadedExpressions.ContainsKey(expressionIndex))
            {
                onError?.Invoke(new ArgumentException($"Expression index {expressionIndex} not available. Available expressions: {string.Join(", ", LoadedExpressions.Keys)}"));
                yield break;
            }

            if (LoadedVoice == null)
            {
                onError?.Invoke(new InvalidOperationException(
                    "Character voice not loaded — the voice folder is missing or the voice failed to design/clone/load; " +
                    "see the earlier error from character creation or load."));
                yield break;
            }

            var liveTalkAPI = LiveTalkAPI.Instance;
            if (liveTalkAPI == null)
            {
                onError?.Invoke(new InvalidOperationException("LiveTalkAPI not initialized. Call LiveTalkAPI.Initialize() first."));
                yield break;
            }

            Logger.LogVerbose($"[Character] {Name} speaking async: \"{text}\" with expression {expressionIndex}");

            AudioClip audioClip = null;
            string cacheKey = null;
            bool audioFromCache = false;

            // Check audio cache first
            if (LiveTalkCache.IsEnabled && !string.IsNullOrEmpty(CharacterId))
            {
                cacheKey = HashUtils.GenerateSpeechCacheKey(text, CharacterId);
                var (exists, cachedPath) = LiveTalkCache.CheckExists(cacheKey);
                
                if (exists)
                {
                    Logger.LogVerbose($"[Character] Loading cached audio for: {text[..Math.Min(30, text.Length)]}...");
                    var loadTask = AudioFileIO.LoadClipAsync(cachedPath);
                    yield return new WaitUntil(() => loadTask.IsCompleted);

                    // A cache hit that cannot be read is not fatal — the line
                    // is regenerated below — but it is not silent either.
                    if (loadTask.IsFaulted)
                    {
                        Logger.LogWarning($"[Character] Cached audio unreadable, regenerating: {cachedPath}: " +
                            loadTask.Exception?.GetBaseException().Message);
                    }
                    else if (loadTask.Result != null)
                    {
                        audioClip = loadTask.Result;
                        audioFromCache = true;
                    }
                }
            }

            // Generate new audio if not cached (with queuing)
            if (audioClip == null)
            {
                // Acquire voice queue lock. The lease is released in the
                // finally below on every exit: success, a fault rethrown by
                // the bridge, or the host stopping this coroutine.
                yield return TaskYield.Wait(liveTalkAPI.VoiceQueue.AcquireAsync(), "Character.VoiceQueue.Acquire");

                try
                {
                    var options = new SpeechOptions { SampleRate = SpeechSampleRate };

                    // Progress<T> captures the SynchronizationContext it is
                    // built on. This coroutine runs on the main thread, so the
                    // engine's worker-thread reports arrive back here rather
                    // than on the thread that generated them — which matters,
                    // because a host will want to hand these to an AudioSource.
                    var chunkRelay = onSpeechChunk == null
                        ? null
                        : new Progress<SpeechChunk>(c => onSpeechChunk(c.Pcm, c.SampleRate));

                    var audioTask = chunkRelay == null
                        ? LoadedVoice.SpeakAsync(text, options)
                        : LoadedVoice.SpeakStreamAsync(text, chunkRelay, options);

                    // A faulted synthesis rethrows here, unwinds through the
                    // finally (releasing the lease) and reaches onError via
                    // the Guard in SpeakAsync.
                    SpeechResult speech = default;
                    yield return TaskYield.Wait(audioTask, r => speech = r, "Character.SpeakAsync (TTS)");

                    // ToAudioClip has to happen here: it is a main-thread API.
                    audioClip = speech.ToAudioClip($"{Name}_speech");
                }
                finally
                {
                    // Release voice queue lock
                    liveTalkAPI.VoiceQueue.Release();
                }
                
                // Save audio to cache (fire and forget)
                if (LiveTalkCache.IsEnabled && !string.IsNullOrEmpty(cacheKey) && audioClip != null)
                {
                    string cachePath = LiveTalkCache.GetFilePath(cacheKey);
                    if (!string.IsNullOrEmpty(cachePath))
                    {
                        var saveTask = AudioFileIO.SaveClipAsync(audioClip, cachePath);
                        _ = saveTask.ContinueWith(t => 
                        {
                            if (t.IsFaulted)
                                Logger.LogWarning($"[Character] Failed to save audio to cache: {t.Exception?.InnerException?.Message}");
                            else
                                Logger.LogVerbose($"[Character] Saved audio to cache: {cacheKey}");
                        });
                    }
                }
            }

            if (audioClip == null)
            {
                onError?.Invoke(new InvalidOperationException(
                    "Generated audio clip is null — the TTS engine returned no audio for this line; see the earlier error."));
                yield break;
            }

            var outputStream = new FrameStream(0);
            
            // For voice-only mode, both callbacks immediately
            if (expressionIndex == -1)
            {
                onAudioReady?.Invoke(outputStream, audioClip);
                onAnimationComplete?.Invoke(outputStream);
                var stopLocal = start.Elapsed;
                Logger.Log($"[Character] Speaking completed for {Name} in {stopLocal.TotalMilliseconds}ms{(audioFromCache ? " (cached)" : "")}");
                yield break;
            }

            // Check for cached animation frames
            if (LiveTalkCache.IsEnabled && !string.IsNullOrEmpty(cacheKey))
            {
                var (framesExist, framesFolder, frameCount) = LiveTalkCache.CheckFramesCacheExists(cacheKey);
                
                if (framesExist && frameCount > 0)
                {
                    Logger.LogVerbose($"[Character] Loading {frameCount} cached animation frames for: {text[..Math.Min(30, text.Length)]}...");
                    
                    // Load frames from cache into output stream
                    outputStream = new FrameStream(frameCount);
                    
                    // Audio ready callback
                    onAudioReady?.Invoke(outputStream, audioClip);
                    
                    // Load frames and call animation complete when done. Guarded
                    // so a failed read finishes the stream and reaches onError
                    // rather than dying inside Unity's coroutine scheduler.
                    var cachedStream = outputStream;
                    liveTalkAPI.Controller.StartCoroutine(TaskYield.Guard(
                        LoadFramesFromCacheWithCallback(framesFolder, frameCount, cachedStream, onAnimationComplete),
                        ex => { cachedStream.Fail(ex); onError?.Invoke(ex); },
                        "Character.LoadFramesFromCache"));
                    
                    var stopCached = start.Elapsed;
                    Logger.Log($"[Character] Audio ready for {Name} in {stopCached.TotalMilliseconds}ms (audio+frames cached, loading...)");
                    yield break;
                }
            }

            // Audio ready - callback immediately, animation will be generated in background
            var expressionData = LoadedExpressions[expressionIndex];
            outputStream = new FrameStream(0); // Will be updated with actual count when generation starts
            
            // Audio ready callback
            onAudioReady?.Invoke(outputStream, audioClip);
            var stopAudio = start.Elapsed;
            Logger.Log($"[Character] Audio ready for {Name} in {stopAudio.TotalMilliseconds}ms{(audioFromCache ? " (cached)" : "")}, animation pending...");

            // Start animation generation in background with queuing. The
            // Guard is what turns a fault inside the animation — a lip-sync
            // model that failed to load, most often — into a finished stream
            // plus an onError call, instead of a consumer that waits forever
            // and a MuseTalk lease that is never given back.
            var animationStream = outputStream;
            liveTalkAPI.Controller.StartCoroutine(TaskYield.Guard(
                GenerateAnimationWithQueue(liveTalkAPI, expressionData.Data, audioClip, animationStream, cacheKey, onAnimationComplete),
                ex => { animationStream.Fail(ex); onError?.Invoke(ex); },
                "Character.GenerateAnimation"));
        }

        /// <summary>
        /// Generate animation frames with queuing to prevent parallel MuseTalk usage.
        /// </summary>
        private static IEnumerator GenerateAnimationWithQueue(
            LiveTalkAPI liveTalkAPI,
            AvatarData avatarData,
            AudioClip audioClip,
            FrameStream outputStream,
            string cacheKey,
            Action<FrameStream> onAnimationComplete)
        {
            // Acquire MuseTalk queue lock. Released in the finally below on
            // every exit — success, fault, or the coroutine being disposed —
            // so a failed animation can never wedge the next one on Acquire.
            yield return TaskYield.Wait(liveTalkAPI.MuseTalkQueue.AcquireAsync(), "Character.MuseTalkQueue.Acquire");

            string framesFolder = null;
            bool completed = false;
            try
            {
                // Generate talking head using MuseTalk with preloaded data
                var generatedStream = liveTalkAPI.GenerateTalkingHeadWithPreloadedData(
                    avatarData,
                    audioClip
                );
                outputStream.TotalExpectedFrames = generatedStream.TotalExpectedFrames;

                // Forward frames from generated stream to output stream
                // If caching is enabled, also save frames
                if (LiveTalkCache.IsEnabled && !string.IsNullOrEmpty(cacheKey))
                {
                    framesFolder = LiveTalkCache.CreateFramesCacheFolder(cacheKey);
                }

                int frameIndex = 0;
                while (generatedStream.HasMoreFrames)
                {
                    var awaiter = generatedStream.WaitForNext();
                    yield return awaiter;

                    if (awaiter.Texture != null)
                    {
                        // Forward frame to output stream
                        outputStream.Queue.Enqueue(awaiter.Texture);

                        // Cache frame if enabled
                        if (!string.IsNullOrEmpty(framesFolder))
                        {
                            byte[] pngData = awaiter.Texture.EncodeToPNG();
                            int currentIndex = frameIndex;
                            _ = Task.Run(() =>
                            {
                                try
                                {
                                    string framePath = Path.Combine(framesFolder, $"frame_{currentIndex:D6}.png");
                                    File.WriteAllBytes(framePath, pngData);
                                }
                                catch (Exception ex)
                                {
                                    Logger.LogWarning($"[Character] Failed to cache frame {currentIndex}: {ex.Message}");
                                }
                            });
                        }

                        frameIndex++;
                    }
                }

                // The producer finishing early because it faulted must not
                // become a short animation that looks complete — and its
                // partial frames folder must not become a cache hit next time.
                if (generatedStream.Error != null)
                {
                    throw new InvalidOperationException(
                        $"Lip-sync animation failed after {frameIndex} frame(s): {generatedStream.Error.Message}",
                        generatedStream.Error);
                }

                outputStream.TotalExpectedFrames = frameIndex;
                outputStream.Finished = true;
                completed = true;
                Logger.LogVerbose($"[Character] Animation generation completed: {frameIndex} frames");
                
                // Animation complete callback
                onAnimationComplete?.Invoke(outputStream);
            }
            finally
            {
                // Consumers draining outputStream exit on every path; the
                // Guard that started this coroutine records the error on it.
                outputStream.Finished = true;

                // A run that faulted or was stopped part-way leaves a short
                // frames folder that the next SpeakAsync would take as a hit.
                if (!completed && framesFolder != null)
                {
                    LiveTalkCache.DeleteFramesCache(cacheKey);
                }

                // Release MuseTalk queue lock
                liveTalkAPI.MuseTalkQueue.Release();
            }
        }

        /// <summary>
        /// Load cached animation frames from disk into a FrameStream with completion callback.
        /// </summary>
        private static IEnumerator LoadFramesFromCacheWithCallback(
            string framesFolder, 
            int frameCount, 
            FrameStream outputStream,
            Action<FrameStream> onAnimationComplete)
        {
            yield return LoadFramesFromCache(framesFolder, frameCount, outputStream);
            onAnimationComplete?.Invoke(outputStream);
        }

        /// <summary>
        /// Load cached animation frames from disk into a FrameStream.
        /// </summary>
        /// <param name="framesFolder">Path to the folder containing cached frames</param>
        /// <param name="frameCount">Number of frames to load</param>
        /// <param name="outputStream">The output stream to populate with frames</param>
        private static IEnumerator LoadFramesFromCache(string framesFolder, int frameCount, FrameStream outputStream)
        {
            outputStream.TotalExpectedFrames = frameCount;

            try
            {
                for (int i = 0; i < frameCount; i++)
                {
                    string framePath = Path.Combine(framesFolder, $"frame_{i:D6}.png");

                    if (!File.Exists(framePath))
                    {
                        Logger.LogWarning($"[Character] Cached frame not found: {framePath}");
                        continue;
                    }

                    // Load frame from disk. A single unreadable cached frame is
                    // skipped (the fault is observed, logged, and the rest of
                    // the clip still plays); the cache entry is best-effort.
                    var loadTask = Task.Run(() => File.ReadAllBytes(framePath));
                    yield return new WaitUntil(() => loadTask.IsCompleted);

                    if (loadTask.IsFaulted)
                    {
                        Logger.LogWarning($"[Character] Failed to load cached frame {i}: {loadTask.Exception?.GetBaseException().Message}");
                        continue;
                    }

                    // Create texture from bytes
                    var texture = new Texture2D(2, 2);
                    if (texture.LoadImage(loadTask.Result))
                    {
                        texture.name = $"cached_frame_{i}";
                        outputStream.Queue.Enqueue(texture);
                    }
                    else
                    {
                        UnityEngine.Object.DestroyImmediate(texture);
                        Logger.LogWarning($"[Character] Failed to decode cached frame {i}");
                    }
                }

                Logger.LogVerbose($"[Character] Loaded {frameCount} frames from cache");
            }
            finally
            {
                outputStream.Finished = true;
            }
        }

        /// <summary>
        /// Load all character data including expressions, voice, and precomputed data.
        /// Throws (so the guarded caller's onError fires) rather than marking a
        /// character loaded without its voice.
        /// </summary>
        internal IEnumerator LoadData()
        {
            Logger.LogVerbose($"[Character] Loading character data for {Name}");

            if (Image != null)
            {
                // Load expressions data
                yield return LoadExpressionsData();
            }

            // Load voice data
            yield return LoadVoiceData();

            if (LoadedVoice == null)
            {
                // LoadVoiceData throws on every failure it can see, so this is
                // the belt to that brace: IsDataLoaded must never be true for
                // a character that cannot speak.
                throw new InvalidOperationException(
                    $"Voice for character '{Name}' did not load from {Path.Combine(CharacterFolder ?? "<no folder>", "voice")}; " +
                    "the character is not usable. See the earlier error for the cause.");
            }

            IsDataLoaded = true;
            Logger.LogVerbose($"[Character] Character data loaded successfully for {Name}");
            
            // Create CharacterPlayer automatically after full load
            CreateCharacterPlayer();
        }

        /// <summary>
        /// Internal coroutine to load character metadata (config + image only, no expressions/voice)
        /// </summary>
        private static IEnumerator LoadCharacterMetadataCoroutine(
            string characterFolder,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            var characterId = Path.GetFileNameWithoutExtension(characterFolder);

            // Load config and image using shared helper
            Character character = null;
            Exception loadError = null;
            
            yield return LoadCharacterConfigAndImageCoroutine(
                characterFolder,
                (c) => character = c,
                (e) => loadError = e);

            if (loadError != null)
            {
                onError?.Invoke(loadError);
                yield break;
            }

            if (character == null)
            {
                onError?.Invoke(new Exception("Failed to load character config and image"));
                yield break;
            }

            character.CharacterId = characterId;
            character.IsDataLoaded = false;  // Mark as NOT fully loaded (metadata only)

            Logger.Log($"[Character] Loaded metadata for {character.Name}");
            onComplete?.Invoke(character);
        }

        /// <summary>
        /// Create avatar internal
        /// </summary>
        /// <param name="useBundle">True to create as macOS bundle, false to create as regular folder</param>
        /// <param name="creationMode">The creation mode to use</param>
        /// <returns>Coroutine for avatar creation</returns>
        private IEnumerator CreateAvatarAsyncInternal(bool useBundle, CreationMode creationMode)
        {
            // Get the LiveTalkAPI instance
            var liveTalkAPI = LiveTalkAPI.Instance ?? throw new InvalidOperationException("LiveTalkAPI not initialized. Call LiveTalkAPI.Initialize() first.");

            // Step 1: Generate a unique ID for this character based on name, gender, voice settings, and image
            CharacterId = HashUtils.GenerateCharacterHash(
                Name,
                Gender.ToString(),
                Pitch.ToString(),
                Speed.ToString(),
                Intro,
                Image,
                VoiceInstruct
            );
            CharacterFolder = Path.Combine(saveLocation, useBundle ? $"{CharacterId}.bundle" : CharacterId);
            // Create main character directory (clean slate approach)
            // Using .bundle extension makes this appear as a single file in macOS Finder
            if (Directory.Exists(CharacterFolder))
            {
                Directory.Delete(CharacterFolder, true);
            }
            Directory.CreateDirectory(CharacterFolder);

            // Add json for character config
            var characterConfig = new
            {
                name = Name,
                gender = Gender,
                pitch = Pitch,
                speed = Speed,
                intro = Intro,
                voiceInstruct = VoiceInstruct
            };
            string characterConfigJson = JsonConvert.SerializeObject(characterConfig, Formatting.Indented);
            string configPath = Path.Combine(CharacterFolder, "character.json");
            yield return TaskYield.Wait(File.WriteAllTextAsync(configPath, characterConfigJson),
                $"Character.CreateAvatar write {configPath}");

            // Add Info.plist for macOS package (only when creating bundle)
            if (useBundle)
            {
                string infoPlistContent = $@"<?xml version=""1.0"" encoding=""UTF-8""?>
<!DOCTYPE plist PUBLIC ""-//Apple//DTD PLIST 1.0//EN"" ""http://www.apple.com/DTDs/PropertyList-1.0.dtd"">
<plist version=""1.0"">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.genesis.livetalk.character.{CharacterId}</string>
    <key>CFBundleName</key>
    <string>{Name}</string>
    <key>CFBundleDisplayName</key>
    <string>{Name} Character</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>BNDL</string>
    <key>CFBundleSignature</key>
    <string>LTCH</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>";
                string plistPath = Path.Combine(CharacterFolder, "Info.plist");
                yield return TaskYield.Wait(File.WriteAllTextAsync(plistPath, infoPlistContent),
                    $"Character.CreateAvatar write {plistPath}");
            }

            // Create subfolder structure
            DrivingFramesFolder = Path.Combine(CharacterFolder, "drivingFrames");
            VoiceFolder = Path.Combine(CharacterFolder, "voice");
            Directory.CreateDirectory(DrivingFramesFolder);
            Directory.CreateDirectory(VoiceFolder);

            Logger.Log($"[Character] Creating character for {Name} in {(useBundle ? "bundle" : "folder")}: {CharacterFolder}");

            if (Image != null)
            {
                // Save image (convert to uncompressed format if needed)
                string imagePath = Path.Combine(CharacterFolder, "image.png");
                var uncompressedImage = TextureUtils.ConvertToUncompressedTexture(Image);
                byte[] imageBytes = uncompressedImage.EncodeToPNG();
                yield return TaskYield.Wait(File.WriteAllBytesAsync(imagePath, imageBytes),
                    $"Character.CreateAvatar write {imagePath}");
                
                // Clean up temporary texture if we created one
                if (uncompressedImage != Image)
                {
                    UnityEngine.Object.DestroyImmediate(uncompressedImage);
                }
                // Step 2: Generate driving frames for each expression
                var expressions = new string[] { "talk-neutral", "approve", "disapprove", "smile", "sad", "surprised", "confused" };
                bool useSingleExpression = creationMode == CreationMode.SingleExpression;
                bool voiceOnly = creationMode == CreationMode.VoiceOnly;
                if (useSingleExpression)
                {
                    expressions = new string[] { "talk-neutral" };
                }
                else if (voiceOnly)
                {
                    expressions = new string[] { };
                }

                // Driving frames depend only on the image and the expression
                // set, so a second character built from the same portrait —
                // a re-rolled or re-locked voice, most often — can take the
                // previous avatar pass instead of spending minutes in
                // LivePortrait again. Copying a few hundred MB costs seconds;
                // regenerating costs minutes.
                string framesCacheKey = expressions.Length == 0
                    ? null
                    : HashUtils.GenerateDrivingFramesCacheKey(
                        imageBytes, creationMode + ":" + string.Join(",", expressions));

                bool framesFromCache = false;
                if (!string.IsNullOrEmpty(framesCacheKey) && LiveTalkCache.IsEnabled)
                {
                    // Frames nest one level down per expression, so this must
                    // be the recursive check — the flat one reports a fully
                    // populated entry as missing.
                    var (cachedExists, cachedFrames) =
                        LiveTalkCache.CheckFolderTreeExists(framesCacheKey, "*.png");
                    if (cachedExists)
                    {
                        Logger.Log("[Character] Reusing cached driving frames for this "
                                   + "portrait — skipping avatar preprocessing.");
                        LiveTalkCache.CopyFolder(cachedFrames, DrivingFramesFolder);
                        framesFromCache = true;
                    }
                }

                for (int expressionIndex = 0;
                     !framesFromCache && expressionIndex < expressions.Length;
                     expressionIndex++)
                {
                    string expression = expressions[expressionIndex];
                    string expressionFolder = Path.Combine(DrivingFramesFolder, $"expression-{expressionIndex}");
                    Directory.CreateDirectory(expressionFolder);

                    Logger.Log($"[Character] Processing expression: {expression} (index: {expressionIndex})");

                    // Load the driving video for this expression
                    VideoClip drivingVideo = LoadDrivingVideoForExpression(expression);
                    if (drivingVideo == null)
                    {
                        Logger.LogWarning($"[Character] Could not load driving video for expression: {expression}");
                        continue;
                    }

                    // Process this expression with coroutines outside try-catch
                    yield return ProcessExpressionCoroutine(expression, drivingVideo, expressionFolder, liveTalkAPI);
                }

                // Populate the cache only after every expression finished. A
                // run that died partway would otherwise leave a short folder
                // that later characters would happily reuse.
                if (!framesFromCache && !string.IsNullOrEmpty(framesCacheKey)
                    && LiveTalkCache.IsEnabled)
                {
                    string cacheFolder = LiveTalkCache.GetFolderPath(framesCacheKey);
                    if (!string.IsNullOrEmpty(cacheFolder))
                    {
                        LiveTalkCache.CopyFolder(DrivingFramesFolder, cacheFolder);
                        Logger.LogVerbose(
                            $"[Character] Cached driving frames: {framesCacheKey}");
                    }
                }
            }
        }

        /// <summary>
        /// Process a single expression with coroutines to handle frame streaming
        /// </summary>
        private IEnumerator ProcessExpressionCoroutine(
            string expression,
            VideoClip drivingVideo, 
            string expressionFolder, 
            LiveTalkAPI liveTalkAPI)
        {
            var videoPlayer = LiveTalkAPI.Instance.Object.GetComponent<VideoPlayer>();
            videoPlayer.clip = drivingVideo;
            videoPlayer.isLooping = false;
            videoPlayer.playOnAwake = false;
            videoPlayer.skipOnDrop = false;
            videoPlayer.Prepare();
            yield return new WaitUntil(() => videoPlayer.isPrepared);

            // Generate animated textures using LivePortrait
            var outputStream = liveTalkAPI.GenerateAnimatedTexturesAsync(Image, videoPlayer);

            // Process frames
            var processResult = new ProcessFramesResult();
            yield return ProcessFramesCoroutine(outputStream, expressionFolder, processResult);
            videoPlayer.clip = null;

            // The LivePortrait producer marks its stream finished on a fault
            // too (so the loop above exits); a truncated expression is a
            // failed character, not a shorter one.
            if (outputStream.Error != null)
            {
                throw new InvalidOperationException(
                    $"Driving-frame generation failed for expression '{expression}': {outputStream.Error.Message}",
                    outputStream.Error);
            }

            Logger.LogVerbose($"[Character] Generated and saved {processResult.GeneratedFrames.Count} frames for expression: {expression}");

            // Generate and save cache data (latents + face data). A fault here
            // — the MuseTalk preprocess models failing to load, usually —
            // rethrows and fails the creation instead of leaving an
            // expression folder without latents.bin / faces.json.
            yield return TaskYield.Wait(GenerateAndSaveCacheData(expressionFolder, processResult),
                $"Character.GenerateAndSaveCacheData {expression}");

            if (LiveTalkAPI.Instance.Config.MemoryUsage == MemoryUsage.Optimal)
            {
                yield return new WaitForSeconds(2f); // Wait for GC to complete
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);
            }
        }

        /// <summary>
        /// Process frame stream using coroutines
        /// </summary>
        private IEnumerator ProcessFramesCoroutine(
            FrameStream outputStream, 
            string expressionFolder,
            ProcessFramesResult result)
        {
            int frameIndex = 0;
            
            // Process frames as they become available using coroutine pattern
            while (outputStream.HasMoreFrames)
            {
                var awaiter = outputStream.WaitForNext();
                yield return awaiter;
                
                if (awaiter.Texture != null)
                {
                    // Save LivePortrait generated frames as numbered PNGs (these are the driving frames)
                    string frameFileName = Path.Combine(expressionFolder, $"{frameIndex:D5}.png");
                    byte[] pngData = awaiter.Texture.EncodeToPNG();
                    yield return TaskYield.Wait(File.WriteAllBytesAsync(frameFileName, pngData),
                        $"Character.ProcessFrames write {frameFileName}");
                    
                    // Keep reference for cache generation
                    if (LiveTalkAPI.Instance.Config.MemoryUsage != MemoryUsage.Optimal)
                    {
                        result.GeneratedFrames.Add(awaiter.Texture);
                    }
                    else
                    {
                        result.GeneratedFramePaths.Add(frameFileName);
                        UnityEngine.Object.DestroyImmediate(awaiter.Texture);
                    }
                    frameIndex++;
                }
            }
        }

        /// <summary>
        /// Load driving video clip for the specified expression
        /// </summary>
        private VideoClip LoadDrivingVideoForExpression(string expression)
        {
            // Try to load from Resources folder
            string[] possiblePaths = new string[]
            {
                $"driving/{expression}",
                $"LiveTalk/driving/{expression}",
                expression
            };

            foreach (string path in possiblePaths)
            {
                var videoClip = Resources.Load<VideoClip>(path);
                if (videoClip != null)
                {
                    Logger.LogVerbose($"[Character] Loaded driving video: {path}");
                    return videoClip;
                }
            }

            Logger.LogWarning($"[Character] Could not find driving video for expression: {expression}");
            return null;
        }

        /// <summary>
        /// Generate and save cache data (latents and face data) for the processed frames using real MuseTalkInference
        /// </summary>
        private async Task GenerateAndSaveCacheData(string expressionFolder, ProcessFramesResult processResult)
        {
            try
            {
                Logger.LogVerbose($"[Character] Generating Cache Data...");

                // Create a temporary MuseTalkInference instance for processing
                var liveTalkAPI = LiveTalkAPI.Instance;
                if (liveTalkAPI == null)
                {
                    Logger.LogError("[Character] LiveTalkAPI not available for cache generation");
                    return;
                }

                // Use MuseTalkInference to process the avatar images and extract real data
                var avatarData = await ProcessAvatarImagesWithMuseTalk(liveTalkAPI, processResult);

                if (avatarData != null && avatarData.Latents.Count > 0)
                {
                    // Save real latents data
                    await SaveLatentsToFile(expressionFolder, avatarData.Latents);

                    // Save real face data
                    await SaveFaceDataToFile(expressionFolder, avatarData.FaceRegions);

                    Logger.LogVerbose($"[Character] Generated real cache data: {avatarData.Latents.Count} latents, {avatarData.FaceRegions.Count} face regions");
                }
                else
                {
                    throw new InvalidOperationException("Failed to generate avatar data using real MuseTalk processing. No fallback available.");
                }
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Character] Error generating real cache data: {ex.Message}");
                throw new InvalidOperationException($"Failed to generate real cache data: {ex.Message}. No fallback available.", ex);
            }
        }

        /// <summary>
        /// Process avatar images using MuseTalkInference public API to extract real latents and face data
        /// This uses the actual MuseTalk face analysis and VAE encoder pipeline - NO FALLBACKS
        /// </summary>
        private async Task<AvatarData> ProcessAvatarImagesWithMuseTalk(LiveTalkAPI liveTalkAPI, ProcessFramesResult processResult)
        {
            Logger.LogVerbose($"[Character] Processing avatar textures using MuseTalk pipeline");

            AvatarData avatarData;
            if (LiveTalkAPI.Instance.Config.MemoryUsage != MemoryUsage.Optimal)
            {
                avatarData = await liveTalkAPI.MuseTalk.ProcessAvatarImages(processResult.GeneratedFrames);
            }
            else
            {
                avatarData = await liveTalkAPI.MuseTalk.ProcessAvatarImages(processResult.GeneratedFramePaths);
            }
            
            if (avatarData?.FaceRegions?.Count == 0 || avatarData?.Latents?.Count == 0)
            {
                throw new InvalidOperationException($"Real MuseTalk processing failed to generate valid avatar data. FaceRegions: {avatarData?.FaceRegions?.Count ?? 0}, Latents: {avatarData?.Latents?.Count ?? 0}");
            }

            Logger.LogVerbose($"[Character] Real MuseTalk processing completed: {avatarData.Latents.Count} latents, {avatarData.FaceRegions.Count} face regions");
            return avatarData;
        }

        /// <summary>
        /// Save real latents data to binary file
        /// </summary>
        private async Task SaveLatentsToFile(string expressionFolder, List<float[]> latents)
        {
            try
            {
                var latentsFile = Path.Combine(expressionFolder, "latents.bin");
                
                // Calculate total size needed
                int totalFloats = latents.Sum(latent => latent.Length);
                var allLatents = new float[totalFloats];
                
                // Combine all latent arrays into one
                int offset = 0;
                foreach (var latent in latents)
                {
                    Array.Copy(latent, 0, allLatents, offset, latent.Length);
                    offset += latent.Length;
                }
                
                // Convert to bytes and save
                var latentsBytes = new byte[allLatents.Length * sizeof(float)];
                Buffer.BlockCopy(allLatents, 0, latentsBytes, 0, latentsBytes.Length);
                await File.WriteAllBytesAsync(latentsFile, latentsBytes);
                
                Logger.LogVerbose($"[Character] Saved {latents.Count} latent arrays ({totalFloats} total floats) to {latentsFile}");
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Character] Error saving latents: {ex.Message}");
            }
        }

        /// <summary>
        /// Save real face data to JSON file and save all precomputed textures
        /// </summary>
        private async Task SaveFaceDataToFile(string expressionFolder, List<FaceData> faceRegions)
        {
            try
            {
                var facesFile = Path.Combine(expressionFolder, "faces.json");
                var texturesFolder = Path.Combine(expressionFolder, "textures");
                
                // Create texture subfolders
                var subfolders = new[]
                {
                    "cropped", "faceLarge", "segmentationMask", "original",
                    "maskSmall", "fullMask", "boundaryMask", "blurredMask"
                };
                
                foreach (var subfolder in subfolders)
                {
                    Directory.CreateDirectory(Path.Combine(texturesFolder, subfolder));
                }

                Logger.LogVerbose($"[Character] Saving face data with precomputed textures for {faceRegions.Count} face regions");

                // Process each face region and save all textures
                var faceDataForJson = new List<object>();
                
                for (int faceIndex = 0; faceIndex < faceRegions.Count; faceIndex++)
                {
                    var face = faceRegions[faceIndex];
                    
                    // Save all precomputed textures for this face
                    var texturePaths = await SaveFaceTextures(texturesFolder, face, faceIndex);
                    
                    // Create face data entry with texture file references
                    var faceDataEntry = new
                    {
                        faceIndex = faceIndex,
                        hasFace = face.HasFace,
                        boundingBox = new
                        {
                            x = face.BoundingBox.x,
                            y = face.BoundingBox.y,
                            width = face.BoundingBox.width,
                            height = face.BoundingBox.height
                        },
                        landmarks = face.Landmarks?.Select(l => new { x = l.x, y = l.y }).ToArray(),
                        adjustedFaceBbox = new
                        {
                            x = face.AdjustedFaceBbox.x,
                            y = face.AdjustedFaceBbox.y,
                            z = face.AdjustedFaceBbox.z,
                            w = face.AdjustedFaceBbox.w
                        },
                        cropBox = new
                        {
                            x = face.CropBox.x,
                            y = face.CropBox.y,
                            z = face.CropBox.z,
                            w = face.CropBox.w
                        },
                        textureDimensions = new
                        {
                            croppedFace = new { width = face.CroppedFaceTexture.width, height = face.CroppedFaceTexture.height },
                            original = new { width = face.OriginalTexture.width, height = face.OriginalTexture.height },
                            faceLarge = new { width = face.FaceLarge.width, height = face.FaceLarge.height },
                            segmentationMask = new { width = face.SegmentationMask.width, height = face.SegmentationMask.height },
                            maskSmall = new { width = face.MaskSmall.width, height = face.MaskSmall.height },
                            fullMask = new { width = face.FullMask.width, height = face.FullMask.height },
                            boundaryMask = new { width = face.BoundaryMask.width, height = face.BoundaryMask.height },
                            blurredMask = new { width = face.BlurredMask.width, height = face.BlurredMask.height }
                        },
                        // Reference to saved texture files
                        textureFiles = texturePaths
                    };
                    
                    faceDataForJson.Add(faceDataEntry);
                }
                
                var faceDataJson = new
                {
                    faceRegions = faceDataForJson.ToArray(),
                    frameCount = faceRegions.Count,
                    timestamp = DateTime.UtcNow,
                    version = "1.0-complete",
                    description = "Complete face data with all precomputed textures saved as PNG files"
                };
                
                string json = JsonConvert.SerializeObject(faceDataJson, Formatting.Indented);
                await File.WriteAllTextAsync(facesFile, json);
                
                Logger.LogVerbose($"[Character] Saved complete face data with textures for {faceRegions.Count} face regions to {facesFile}");
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Character] Error saving face data with textures: {ex.Message}");
            }
        }

        /// <summary>
        /// Save all precomputed textures for a single face region
        /// </summary>
        private async Task<Dictionary<string, string>> SaveFaceTextures(string texturesFolder, FaceData face, int faceIndex)
        {
            var texturePaths = new Dictionary<string, string>();
            
            try
            {
                // Define texture mappings: texture data -> folder name -> filename
                // Note: Removed "original" to eliminate redundancy - driving frames are saved as numbered PNGs
                var textureMap = new List<(Frame frame, string folder, string key)>
                {
                    (face.CroppedFaceTexture, "cropped", "croppedFace"),
                    (face.FaceLarge, "faceLarge", "faceLarge"),
                    (face.SegmentationMask, "segmentationMask", "segmentationMask"),
                    (face.OriginalTexture, "original", "original"),
                    (face.MaskSmall, "maskSmall", "maskSmall"),
                    (face.FullMask, "fullMask", "fullMask"),
                    (face.BoundaryMask, "boundaryMask", "boundaryMask"),
                    (face.BlurredMask, "blurredMask", "blurredMask")
                };

                foreach (var (frame, folder, key) in textureMap)
                {
                    if (frame.data != null && frame.data.Length > 0)
                    {
                        string filename = $"face_{faceIndex:D3}.bytes";
                        string folderPath = Path.Combine(texturesFolder, folder);
                        string fullPath = Path.Combine(folderPath, filename);
                        
                        // Save as bytes array
                        await File.WriteAllBytesAsync(fullPath, frame.data);
                        
                        // Store relative path for JSON reference
                        string relativePath = Path.Combine("textures", folder, filename).Replace('\\', '/');
                        texturePaths[key] = relativePath;
                        
                        Logger.LogVerbose($"[Character] Saved {key} texture: {relativePath} ({frame.width}x{frame.height})");
                    }
                    else
                    {
                        texturePaths[key] = null; // Mark as missing/empty
                        Logger.LogWarning($"[Character] {key} texture data is null or empty for face {faceIndex}");
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Character] Error saving textures for face {faceIndex}: {ex.Message}");
            }
            
            return texturePaths;
        }

        /// <summary>PCM view of a clip, so it can be saved as a voice sample.</summary>
        private static SpeechResult ClipToSpeechResult(AudioClip clip)
        {
            var interleaved = new float[clip.samples * clip.channels];
            clip.GetData(interleaved, 0);
            if (clip.channels <= 1)
                return new SpeechResult(interleaved, clip.frequency);

            var mono = new float[clip.samples];
            for (int i = 0; i < clip.samples; i++)
            {
                float sum = 0f;
                for (int c = 0; c < clip.channels; c++)
                    sum += interleaved[i * clip.channels + c];
                mono[i] = sum / clip.channels;
            }
            return new SpeechResult(mono, clip.frequency);
        }

        private async Task LoadVoiceFromReference(string voicePromptPath, string voiceFolder)
        {
            // Thrown, not logged-and-returned: a silent return here left
            // LoadedVoice null and the character reported as created, with
            // the failure only showing up later as "Character voice not
            // loaded". The bridge in CreateAvatarCore logs and rethrows.
            var voicePromptClip = await AudioFileIO.LoadClipAsync(voicePromptPath)
                ?? throw new FileNotFoundException(
                    $"Could not read the voice prompt (clone reference) at {voicePromptPath}", voicePromptPath);

            // Async because a cold clone loads the Base tables plus two
            // reference encoders - tens of seconds of main-thread stall
            // otherwise.
            var voice = await QwenTts.CreateClonedVoiceAsync(voicePromptClip, VoiceCloneRefText)
                ?? throw new InvalidOperationException(
                    $"Failed to clone the character voice from the reference {voicePromptPath} — the TTS engine returned no voice.");

            // The reference recording *is* this voice's sample, so it is saved
            // as one. SaveAsync also stores the derived clone prompt, which is
            // what stops the next load re-running both encoders.
            var referenceSample = ClipToSpeechResult(voicePromptClip);
            await voice.SaveAsync(voiceFolder, referenceSample);
            LoadedVoice = voice;
            VoiceSampleClip = voicePromptClip;
        }

        /// <summary>
        /// Generate a designed voice and a rendered sample for it.
        /// Voice samples are cached based on style parameters and characterId (characterId, gender, pitch, speed, intro).
        /// </summary>
        private async Task GenerateVoiceSample(string voiceFolder)
        {
            string cacheKey = HashUtils.GenerateVoiceStyleCacheKey(
                CharacterId, Gender.ToString(), Pitch.ToString(), Speed.ToString(), Intro, VoiceInstruct);

            // Check cache first
            if (LiveTalkCache.IsEnabled)
            {
                var (exists, cachedFolder) = LiveTalkCache.CheckFolderExists(cacheKey);                
                if (exists)
                {
                    Logger.Log($"[Character] Using cached voice sample for style: {Gender}/{Pitch}/{Speed}");
                    LiveTalkCache.CopyFolder(cachedFolder, voiceFolder);
                    return;
                }
            }

            Logger.LogVerbose($"[Character] Generating voice sample: Gender={Gender}, Pitch={Pitch}, Speed={Speed}");

            // Thrown, not logged: see LoadVoiceFromReference.
            var voice = await QwenTts.CreateDesignedVoiceAsync(
                new VoiceDesignSpec(LiveTalk.Utils.VoiceInstruct.Compose(Gender, Pitch, Speed, VoiceInstruct)))
                ?? throw new InvalidOperationException(
                    $"Failed to design the voice for '{Name}' ({Gender}/{Pitch}/{Speed}) — the TTS engine returned no voice. " +
                    "Check that the VoiceDesign model is present and initialized.");

            // Render the intro so the voice has a sample. A designed voice
            // has no inherent audio, and callers downstream (voice preview,
            // and cloning a chosen take) need a wav on disk. Without this
            // the folder claims a sample it does not have.
            var sample = await voice.SpeakAsync(Intro);
            await voice.SaveAsync(voiceFolder, sample);
            LoadedVoice = voice;
            VoiceSampleClip = sample.ToAudioClip($"{Name}_sample");

            // Save to cache
            if (LiveTalkCache.IsEnabled)
            {
                string cacheFolder = LiveTalkCache.GetFolderPath(cacheKey);
                if (!string.IsNullOrEmpty(cacheFolder))
                {
                    LiveTalkCache.CopyFolder(voiceFolder, cacheFolder);
                    Logger.LogVerbose($"[Character] Saved voice sample to cache: {cacheKey}");
                }
            }
        }

        /// <summary>
        /// Load all expression data (frames, latents, face data)
        /// </summary>
        private IEnumerator LoadExpressionsData()
        {   
            string drivingFramesFolder = Path.Combine(CharacterFolder, "drivingFrames");
            if (!Directory.Exists(drivingFramesFolder))
            {
                Logger.LogWarning($"[Character] No driving frames folder found: {drivingFramesFolder}");
                yield break;
            }

            var expressionFolders = Directory.GetDirectories(drivingFramesFolder);
            Logger.LogVerbose($"[Character] Found {expressionFolders.Length} expression folders");

            for (int i = 0; i < expressionFolders.Length; i++)
            {
                string expressionFolder = expressionFolders[i];
                string folderName = Path.GetFileName(expressionFolder);
                
                // Extract expression index from folder name (expression-0, expression-1, etc.)
                if (folderName.StartsWith("expression-") && int.TryParse(folderName[11..], out int expressionIndex))
                {
                    var expressionData = new ExpressionData
                    {
                        ExpressionName = GetExpressionName(expressionIndex)
                    };

                    // Load latents
                    yield return LoadExpressionLatents(expressionFolder, expressionData);

                    // Load face data
                    yield return LoadExpressionFaceData(expressionFolder, expressionData);

                    LoadedExpressions[expressionIndex] = expressionData;
                    Logger.LogVerbose($"[Character] Loaded expression {expressionIndex} ({expressionData.ExpressionName}): {expressionData.Data.FaceRegions.Count} frames");
                }
            }
        }

        /// <summary>
        /// Load latents for a specific expression - optimized with unsafe code and parallelization
        /// </summary>
        private static IEnumerator LoadExpressionLatents(string expressionFolder, ExpressionData expressionData)
        {
            string latentsFile = Path.Combine(expressionFolder, "latents.bin");
            if (!File.Exists(latentsFile))
            {
                Logger.LogWarning($"[Character] No latents file found: {latentsFile}");
                yield break;
            }

            // Both steps rethrow on fault. Skipping them used to leave the
            // expression with zero latents, which surfaced much later as
            // "No avatar latents available" on the first animated line.
            byte[] latentsBytes = null;
            yield return TaskYield.Wait(File.ReadAllBytesAsync(latentsFile), b => latentsBytes = b,
                $"Character.LoadExpressionLatents read {latentsFile}");

            // Process latents in parallel using unsafe code for optimal performance
            yield return TaskYield.Wait(Task.Run(() => ProcessLatentsUnsafe(latentsBytes, expressionData)),
                $"Character.LoadExpressionLatents process {latentsFile}");
        }

        /// <summary>
        /// Process latents using unsafe code and parallel processing for maximum performance
        /// </summary>
        private static unsafe void ProcessLatentsUnsafe(byte[] latentsBytes, ExpressionData expressionData)
        {
            const int latentSize = 8 * 32 * 32; // 8192 floats per latent
            const int floatSize = sizeof(float);
            const int latentSizeBytes = latentSize * floatSize;
            
            int totalFloats = latentsBytes.Length / floatSize;
            int numLatents = totalFloats / latentSize;
            
            if (numLatents == 0)
            {
                Logger.LogWarning("[Character] No valid latents found in file");
                return;
            }

            // Pre-allocate list capacity to avoid resizing
            expressionData.Data.Latents.Capacity = numLatents;
            
            // Create all latent arrays upfront to avoid allocations in parallel loop
            var latentArrays = new float[numLatents][];
            for (int i = 0; i < numLatents; i++)
            {
                latentArrays[i] = new float[latentSize];
            }

            // Pin the source bytes for unsafe access
            fixed (byte* sourcePtr = latentsBytes)
            {
                float* floatPtr = (float*)sourcePtr;
                
                // Process latents in parallel with optimal memory access
                System.Threading.Tasks.Parallel.For(0, numLatents, new ParallelOptions
                {
                    MaxDegreeOfParallelism = Environment.ProcessorCount
                }, latentIndex =>
                {
                    var targetArray = latentArrays[latentIndex];
                    float* sourceLatentPtr = floatPtr + (latentIndex * latentSize);
                    
                    // Pin target array for direct memory copy
                    fixed (float* targetPtr = targetArray)
                    {
                        // Direct memory copy - much faster than Array.Copy or Buffer.BlockCopy
                        Buffer.MemoryCopy(sourceLatentPtr, targetPtr, latentSizeBytes, latentSizeBytes);
                    }
                });
            }
            
            // Add all processed latents to the expression data
            // This is done sequentially to avoid thread safety issues with List<T>
            for (int i = 0; i < numLatents; i++)
            {
                expressionData.Data.Latents.Add(latentArrays[i]);
            }
        }

        /// <summary>
        /// Load face data for a specific expression
        /// </summary>
        private static IEnumerator LoadExpressionFaceData(string expressionFolder, ExpressionData expressionData)
        {
            string facesFile = Path.Combine(expressionFolder, "faces.json");
            if (!File.Exists(facesFile))
            {
                Logger.LogWarning($"[Character] No faces file found: {facesFile}");
                yield break;
            }

            string facesJson = null;
            yield return TaskYield.Wait(File.ReadAllTextAsync(facesFile), t => facesJson = t,
                $"Character.LoadExpressionFaceData read {facesFile}");

            yield return TaskYield.Wait(ParseFaceDataJson(facesJson, expressionData, expressionFolder),
                $"Character.LoadExpressionFaceData parse {facesFile}");
        }

        /// <summary>
        /// Load voice data for the character from the saved voice folder.
        /// Throws on a missing folder, a missing <c>voice.json</c>, or a
        /// faulted engine load: a character without a voice is not loaded.
        /// </summary>
        private IEnumerator LoadVoiceData()
        {
            string voiceFolder = Path.Combine(CharacterFolder, "voice");
            if (!Directory.Exists(voiceFolder))
            {
                throw new DirectoryNotFoundException(
                    $"No voice folder for character '{Name}': {voiceFolder}. " +
                    "The character was never given a voice, or its creation failed part-way.");
            }

            string voiceJson = Path.Combine(voiceFolder, "voice.json");
            if (!File.Exists(voiceJson))
            {
                throw new FileNotFoundException(
                    $"Voice folder for character '{Name}' has no voice.json: {voiceJson}. " +
                    "Voice design or clone did not complete; recreate the character.", voiceJson);
            }

            // Restores the stored clone prompt when there is one, so this does
            // not re-run the speaker and tokenizer encoders. A fault (engine
            // not initialized, corrupt prompt, missing reference) rethrows.
            QwenVoice voice = null;
            yield return TaskYield.Wait(QwenTts.LoadVoiceAsync(voiceFolder), v => voice = v,
                $"Character.LoadVoiceData {voiceFolder}");

            LoadedVoice = voice ?? throw new InvalidOperationException(
                $"The TTS engine returned no voice for {voiceFolder}.");
            Logger.LogVerbose($"[Character] Voice loaded from folder for {Name}");

            // The rendered sample is optional; a designed voice only has one
            // if a take was saved with it. An unreadable one is logged and the
            // character still loads.
            string samplePath = Path.Combine(voiceFolder, "sample.wav");
            if (File.Exists(samplePath))
            {
                var sampleTask = AudioFileIO.LoadClipAsync(samplePath);
                yield return new WaitUntil(() => sampleTask.IsCompleted);
                if (sampleTask.IsFaulted)
                    Logger.LogWarning($"[Character] Voice sample unreadable, continuing without it: {samplePath}: " +
                        sampleTask.Exception?.GetBaseException().Message);
                else
                    VoiceSampleClip = sampleTask.Result;
            }
        }

        /// <summary>
        /// Get expression name from index
        /// </summary>
        private static string GetExpressionName(int index)
        {
            var expressions = new string[] { "talk-neutral", "approve", "disapprove", "smile", "sad", "surprised", "confused" };
            return index < expressions.Length ? expressions[index] : $"expression-{index}";
        }

        /// <summary>
        /// Parse face data JSON and load all associated textures
        /// </summary>
        private static async Task ParseFaceDataJson(string facesJson, ExpressionData expressionData, string expressionFolder)
        {
            try
            {
                // Parse the JSON using a proper data structure instead of dynamic
                var faceDataJson = JsonConvert.DeserializeObject<FaceDataContainer>(facesJson);
                
                if (faceDataJson?.faceRegions != null)
                {
                    var tasks = new List<Task>();
                    for (int i = 0; i < faceDataJson.faceRegions.Length; i++)
                    {
                        var faceRegion = faceDataJson.faceRegions[i];
                        // Create complete face data structure with all loaded textures
                        var faceData = new FaceData
                        {
                            HasFace = faceRegion.hasFace,
                            BoundingBox = new Rect(
                                faceRegion.boundingBox.x,
                                faceRegion.boundingBox.y,
                                faceRegion.boundingBox.width,
                                faceRegion.boundingBox.height
                            ),  
                            AdjustedFaceBbox = new Vector4(
                                faceRegion.adjustedFaceBbox.x,
                                faceRegion.adjustedFaceBbox.y,
                                faceRegion.adjustedFaceBbox.z,
                                faceRegion.adjustedFaceBbox.w
                            ),
                            CropBox = new Vector4(
                                faceRegion.cropBox.x,
                                faceRegion.cropBox.y,
                                faceRegion.cropBox.z,
                                faceRegion.cropBox.w
                            )   
                        };
                        expressionData.Data.FaceRegions.Add(faceData);
                        tasks.Add(LoadFaceTextures(faceData, faceRegion, expressionFolder));
                    }
                    await Task.WhenAll(tasks);
                }
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Character] Error parsing face data: {ex.Message}");
            }
        }

        /// <summary>
        /// Load all face textures from saved files
        /// </summary>
        private static async Task LoadFaceTextures(FaceData faceData, FaceRegionData faceRegion, string expressionFolder)
        {
            try
            {
                // Define texture mappings to eliminate code duplication
                var textureLoaders = new[]
                {
                    new { 
                        FilePath = faceRegion.textureFiles?.croppedFace,
                        Dimensions = faceRegion.textureDimensions.croppedFace,
                        SetTexture = new Action<Frame>(frame => faceData.CroppedFaceTexture = frame)
                    },
                    new { 
                        FilePath = faceRegion.textureFiles?.faceLarge,
                        Dimensions = faceRegion.textureDimensions.faceLarge,
                        SetTexture = new Action<Frame>(frame => faceData.FaceLarge = frame)
                    },
                    new { 
                        FilePath = faceRegion.textureFiles?.segmentationMask,
                        Dimensions = faceRegion.textureDimensions.segmentationMask,
                        SetTexture = new Action<Frame>(frame => faceData.SegmentationMask = frame)
                    },
                    new { 
                        FilePath = faceRegion.textureFiles?.maskSmall,
                        Dimensions = faceRegion.textureDimensions.maskSmall,
                        SetTexture = new Action<Frame>(frame => faceData.MaskSmall = frame)
                    },
                    new { 
                        FilePath = faceRegion.textureFiles?.original,
                        Dimensions = faceRegion.textureDimensions.original,
                        SetTexture = new Action<Frame>(frame => faceData.OriginalTexture = frame)
                    },
                    new { 
                        FilePath = faceRegion.textureFiles?.fullMask,
                        Dimensions = faceRegion.textureDimensions.fullMask,
                        SetTexture = new Action<Frame>(frame => faceData.FullMask = frame)
                    },
                    new { 
                        FilePath = faceRegion.textureFiles?.boundaryMask,
                        Dimensions = faceRegion.textureDimensions.boundaryMask,
                        SetTexture = new Action<Frame>(frame => faceData.BoundaryMask = frame)
                    },
                    new { 
                        FilePath = faceRegion.textureFiles?.blurredMask,
                        Dimensions = faceRegion.textureDimensions.blurredMask,
                        SetTexture = new Action<Frame>(frame => faceData.BlurredMask = frame)
                    }
                };

                var tasks = textureLoaders
                    .Where(loader => !string.IsNullOrEmpty(loader.FilePath))
                    .Select(loader => Task.Run(async () =>
                    {
                        string texturePath = Path.Combine(expressionFolder, loader.FilePath);
                        var frame = await LoadTextureAsFrame(texturePath, loader.Dimensions.width, loader.Dimensions.height);
                        loader.SetTexture(frame);
                    }))
                    .ToList();

                await Task.WhenAll(tasks);
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Character] Error loading face textures: {ex.Message}");
            }
        }

        /// <summary>
        /// Load a texture file and convert it to Frame format
        /// </summary>
        private static async Task<Frame> LoadTextureAsFrame(string texturePath, int width, int height)
        {
            try
            {
                if (!File.Exists(texturePath))
                {
                    Logger.LogWarning($"[Character] Texture file not found: {texturePath}");
                    return new Frame(); // Return empty frame
                }

                return new Frame(await File.ReadAllBytesAsync(texturePath), width, height);
            }
            catch (Exception ex)
            {
                Logger.LogError($"[Character] Error loading texture {texturePath}: {ex.Message}");
                return new Frame();
            }
        }

        /// <summary>
        /// Shared helper to load character config JSON and image
        /// </summary>
        private static IEnumerator LoadCharacterConfigAndImageCoroutine(
            string characterFolder,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            // Load character.json
            string configPath = Path.Combine(characterFolder, "character.json");
            if (!File.Exists(configPath))
            {
                onError?.Invoke(new FileNotFoundException($"Character config file not found: {configPath}"));
                yield break;
            }

            var readConfigTask = File.ReadAllTextAsync(configPath);
            yield return new WaitUntil(() => readConfigTask.IsCompleted);

            if (readConfigTask.IsFaulted)
            {
                onError?.Invoke(readConfigTask.Exception?.InnerException ?? new Exception("Failed to read character config"));
                yield break;
            }

            // Parse character config
            CharacterConfig config;
            try
            {
                config = JsonConvert.DeserializeObject<CharacterConfig>(readConfigTask.Result);
            }
            catch (Exception ex)
            {
                onError?.Invoke(new Exception($"Failed to parse character config: {ex.Message}"));
                yield break;
            }

            // Load character image
            string imagePath = Path.Combine(characterFolder, "image.png");
            Texture2D texture = null;
            if (!File.Exists(imagePath))
            {
                Logger.Log($"[Character] {config.name} image not found: {imagePath}");
            }
            else
            {
                var readImageTask = File.ReadAllBytesAsync(imagePath);
                yield return new WaitUntil(() => readImageTask.IsCompleted);

                if (readImageTask.IsFaulted)
                {
                    onError?.Invoke(readImageTask.Exception?.InnerException ?? new Exception("Failed to read character image"));
                    yield break;
                }

                // Create texture from image bytes
                var imageBytes = readImageTask.Result;
                texture = new Texture2D(2, 2); // Temporary size, will be replaced by LoadImage
                if (!texture.LoadImage(imageBytes))
                {
                    onError?.Invoke(new Exception("Failed to load character image into texture"));
                    yield break;
                }
            }

            // Create character object with config and image
            var character = new Character(
                config.name,
                config.gender,
                texture,
                config.pitch,
                config.speed,
                config.intro
            )
            {
                CharacterFolder = characterFolder,
                VoiceInstruct = config.voiceInstruct
            };

            onComplete?.Invoke(character);
        }

        /// <summary>
        /// Load character data from the character folder or bundle (full load with expressions/voice)
        /// </summary>
        /// <param name="characterFolder">The folder or bundle containing the character data</param>
        /// <param name="onComplete">Callback when character data is successfully loaded</param>
        /// <param name="onError">Callback when an error occurs</param>
        private static IEnumerator LoadCharacterDataCoroutine(
            string characterFolder,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            var start = System.Diagnostics.Stopwatch.StartNew();
            var characterId = Path.GetFileNameWithoutExtension(characterFolder);

            // Load config and image using shared helper
            Character character = null;
            Exception loadError = null;
            
            yield return LoadCharacterConfigAndImageCoroutine(
                characterFolder,
                (c) => character = c,
                (e) => loadError = e);

            if (loadError != null)
            {
                onError?.Invoke(loadError);
                yield break;
            }

            if (character == null)
            {
                onError?.Invoke(new Exception("Failed to load character config and image"));
                yield break;
            }

            character.CharacterId = characterId;

            // Load all character data (expressions, voice, etc.)
            yield return character.LoadData();
            var elapsed = start.Elapsed;
            bool isBundle = characterFolder.EndsWith(".bundle");
            Logger.LogVerbose($"[Character] Character data for {character.Name} loaded from {(isBundle ? "bundle" : "folder")} in {elapsed.TotalMilliseconds} milliseconds");
            onComplete?.Invoke(character);
        }

        /// <summary>
        /// Get the full path to a character by ID, supporting both folder and .bundle package formats
        /// </summary>
        /// <param name="characterId">The character ID to find</param>
        /// <returns>The full path to the character folder/bundle, or null if not found</returns>
        private static string GetCharacterPath(string characterId)
        {
            if (string.IsNullOrEmpty(saveLocation) || string.IsNullOrEmpty(characterId))
            {
                return null;
            }

            // Check for .bundle package first (macOS format)
            string bundlePath = Path.Combine(saveLocation, $"{characterId}.bundle");
            if (Directory.Exists(bundlePath) && File.Exists(Path.Combine(bundlePath, "character.json")))
            {
                return bundlePath;
            }

            // Check for regular folder (universal format)
            string folderPath = Path.Combine(saveLocation, characterId);
            if (Directory.Exists(folderPath) && File.Exists(Path.Combine(folderPath, "character.json")))
            {
                return folderPath;
            }

            return null;
        }

        /// <summary>
        /// Check if a character exists as a bundle package
        /// </summary>
        /// <param name="characterId">The character ID to check</param>
        /// <returns>True if the character exists as a .bundle package</returns>
        private static bool IsCharacterBundle(string characterId)
        {
            if ( string.IsNullOrEmpty(saveLocation) || string.IsNullOrEmpty(characterId))
            {
                return false;
            }

            string bundlePath = Path.Combine(saveLocation, $"{characterId}.bundle");
            return Directory.Exists(bundlePath) && File.Exists(Path.Combine(bundlePath, "character.json"));
        }

        /// <summary>
        /// Check if a character exists as a regular folder
        /// </summary>
        /// <param name="characterId">The character ID to check</param>
        /// <returns>True if the character exists as a regular folder</returns>
        private static bool IsCharacterFolder(string characterId)
        {
            if (string.IsNullOrEmpty(saveLocation) || string.IsNullOrEmpty(characterId))
            {
                return false;
            }

            string folderPath = Path.Combine(saveLocation, characterId);
            return Directory.Exists(folderPath) && File.Exists(Path.Combine(folderPath, "character.json"));
        }

        /// <summary>
        /// Get the format type of a character
        /// </summary>
        /// <param name="characterId">The character ID to check</param>
        /// <returns>The format type: "bundle", "folder", or null if not found</returns>
        private static string GetCharacterFormat(string characterId)
        {
            if (IsCharacterBundle(characterId)) return "bundle";
            if (IsCharacterFolder(characterId)) return "folder";
            return null;
        }
        
        /// <summary>
        /// Creates and initializes the CharacterPlayer for this character
        /// </summary>
        private void CreateCharacterPlayer()
        {
            if (_characterPlayer != null || !IsDataLoaded)
                return;
            
            // Create GameObject for CharacterPlayer
            var playerObject = new GameObject($"CharacterPlayer_{Name}");
            playerObject.transform.SetParent(CharacterPlayer.ParentTransform);
            
            // Add CharacterPlayer component
            _characterPlayer = playerObject.AddComponent<CharacterPlayer>();
            
            // Assign this character to the player
            _characterPlayer.AssignCharacter(this);
            
            Logger.Log($"[LiveTalk.Character] Created CharacterPlayer for {Name}");
        }
    }

}
