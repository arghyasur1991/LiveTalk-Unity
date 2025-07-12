using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Video;
using SparkTTS;
using SparkTTS.Utils;
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
        public Texture2D Image { get; internal set; }
        public Pitch Pitch { get; internal set; }
        public Speed Speed { get; internal set; }
        public string Intro { get; internal set; } = "Hello, this is a test message";
        internal static string saveLocation;
        
        // Loaded character data for inference
        public bool IsDataLoaded { get; internal set; } = false;
        internal string CharacterFolder { get; set; }
        internal Dictionary<int, ExpressionData> LoadedExpressions { get; set; } = new Dictionary<int, ExpressionData>();
        internal CharacterVoice LoadedVoice { get; set; }
        
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

        public static IEnumerator LoadCharacterAsync(
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

            Character loadedCharacter = null;
            Exception loadError = null;

            bool isBundle = IsCharacterBundle(characterId);
            Logger.Log($"[Character] Loading character data for {characterId} from {(isBundle ? "bundle" : "folder")}: {characterPath}");

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

        /// <summary>
        /// Create avatar with explicit format selection
        /// </summary>
        /// <param name="useBundle">True to create as macOS bundle, false to create as regular folder</param>
        public IEnumerator CreateAvatarAsync(bool useBundle, CreationMode creationMode)
        {
            var start = System.Diagnostics.Stopwatch.StartNew();
            // Get the LiveTalkAPI instance
            var liveTalkAPI = LiveTalkAPI.Instance ?? throw new InvalidOperationException("LiveTalkAPI not initialized. Call LiveTalkAPI.Initialize() first.");

            // Step 1: Generate a unique ID for this character based on name, gender, and image
            string characterId = GenerateCharacterHash();
            string characterFolder = Path.Combine(saveLocation, useBundle ? $"{characterId}.bundle" : characterId);
            
            // Create main character directory (clean slate approach)
            // Using .bundle extension makes this appear as a single file in macOS Finder
            if (Directory.Exists(characterFolder))
            {
                Directory.Delete(characterFolder, true);
            }
            Directory.CreateDirectory(characterFolder);

            // Add json for character config
            var characterConfig = new
            {
                name = Name,
                gender = Gender,
                pitch = Pitch,
                speed = Speed, 
                intro = Intro
            };
            string characterConfigJson = JsonConvert.SerializeObject(characterConfig, Formatting.Indented);
            var writeConfigTask = File.WriteAllTextAsync(Path.Combine(characterFolder, "character.json"), characterConfigJson);
            yield return new WaitUntil(() => writeConfigTask.IsCompleted);

            // Add Info.plist for macOS package (only when creating bundle)
            if (useBundle)
            {
                string infoPlistContent = $@"<?xml version=""1.0"" encoding=""UTF-8""?>
<!DOCTYPE plist PUBLIC ""-//Apple//DTD PLIST 1.0//EN"" ""http://www.apple.com/DTDs/PropertyList-1.0.dtd"">
<plist version=""1.0"">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.genesis.livetalk.character.{characterId}</string>
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
                var writePlistTask = File.WriteAllTextAsync(Path.Combine(characterFolder, "Info.plist"), infoPlistContent);
                yield return new WaitUntil(() => writePlistTask.IsCompleted);
            }

            // Save image (convert to uncompressed format if needed)
            string imagePath = Path.Combine(characterFolder, "image.png");
            var uncompressedImage = TextureUtils.ConvertToUncompressedTexture(Image);
            byte[] imageBytes = uncompressedImage.EncodeToPNG();
            var writeImageTask = File.WriteAllBytesAsync(imagePath, imageBytes);
            yield return new WaitUntil(() => writeImageTask.IsCompleted);
            
            // Clean up temporary texture if we created one
            if (uncompressedImage != Image)
            {
                UnityEngine.Object.DestroyImmediate(uncompressedImage);
            }
            
            // Create subfolder structure
            string drivingFramesFolder = Path.Combine(characterFolder, "drivingFrames");
            string voiceFolder = Path.Combine(characterFolder, "voice");
            Directory.CreateDirectory(drivingFramesFolder);
            Directory.CreateDirectory(voiceFolder);

            Logger.Log($"[Character] Creating character for {Name} in {(useBundle ? "bundle" : "folder")}: {characterFolder}");

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
            for (int expressionIndex = 0; expressionIndex < expressions.Length; expressionIndex++)
            {
                string expression = expressions[expressionIndex];
                string expressionFolder = Path.Combine(drivingFramesFolder, $"expression-{expressionIndex}");
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

            // Step 3: Generate voice sample using SparkTTS
            var voiceTask = GenerateVoiceSample(voiceFolder);
            yield return new WaitUntil(() => voiceTask.IsCompleted);

            var stop = start.Elapsed;
            Logger.Log($"[Character] Character creation completed for {Name} in {stop.TotalMilliseconds}ms");
        }

        /// <summary>
        /// Generate speech asynchronously using coroutines
        /// </summary>
        /// <param name="text">Text to speak</param>
        /// <param name="expressionIndex">Expression to use, -1 for voice only</param>
        /// <param name="onComplete">Callback when audio generation is complete</param>
        /// <param name="onError">Callback when an error occurs</param>
        /// <returns>Coroutine for audio generation, then FrameStream for video frames</returns>
        public IEnumerator SpeakAsync(
            string text, 
            int expressionIndex = 0,
            Action<FrameStream, AudioClip> onComplete = null,
            Action<Exception> onError = null)
        {
            var start = System.Diagnostics.Stopwatch.StartNew();
            if (!IsDataLoaded)
            {
                onError?.Invoke(new InvalidOperationException("Character data not loaded. Use CharacterFactory.LoadCharacterAsync() first."));
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
                onError?.Invoke(new InvalidOperationException("Character voice not loaded."));
                yield break;
            }

            var liveTalkAPI = LiveTalkAPI.Instance;
            if (liveTalkAPI == null)
            {
                onError?.Invoke(new InvalidOperationException("CharacterFactory not initialized. Call CharacterFactory.Initialize() first."));
                yield break;
            }

            Logger.LogVerbose($"[Character] {Name} speaking async: \"{text}\" with expression {expressionIndex}");

            // Generate audio using the loaded character voice
            var audioTask = LoadedVoice.GenerateSpeechAsync(text);
            yield return new WaitUntil(() => audioTask.IsCompleted);

            if (audioTask.IsFaulted)
            {
                onError?.Invoke(audioTask.Exception?.InnerException ?? new Exception("Failed to generate speech audio."));
                yield break;
            }

            var audioClip = audioTask.Result;
            if (audioClip == null)
            {
                onError?.Invoke(new InvalidOperationException("Generated audio clip is null."));
                yield break;
            }
            var outputStream = new FrameStream(0);
            if (expressionIndex == -1)
            {
                onComplete?.Invoke(outputStream, audioClip);
                var stopLocal = start.Elapsed;
                Logger.Log($"[Character] Speaking completed for {Name} in {stopLocal.TotalMilliseconds}ms");
                yield break;
            }

            // Use the preloaded expression data for MuseTalk
            var expressionData = LoadedExpressions[expressionIndex];
            
            // Generate talking head using MuseTalk with preloaded data
            outputStream = liveTalkAPI.GenerateTalkingHeadWithPreloadedData(
                expressionData.Data,
                audioClip
            );

            onComplete?.Invoke(outputStream, audioClip);
            var stop = start.Elapsed;
            Logger.Log($"[Character] Speaking completed for {Name} in {stop.TotalMilliseconds}ms");
        }

        /// <summary>
        /// Load all character data including expressions, voice, and precomputed data
        /// </summary>
        internal IEnumerator LoadData()
        {
            Logger.LogVerbose($"[Character] Loading character data for {Name}");

            // Load expressions data
            yield return LoadExpressionsData();

            // Load voice data
            yield return LoadVoiceData();

            IsDataLoaded = true;
            Logger.LogVerbose($"[Character] Character data loaded successfully for {Name}");
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

            Logger.LogVerbose($"[Character] Generated and saved {processResult.GeneratedFrames.Count} frames for expression: {expression}");

            // Generate and save cache data
            var cacheTask = GenerateAndSaveCacheData(expressionFolder, processResult);
            yield return new WaitUntil(() => cacheTask.IsCompleted);

            if (LiveTalkAPI.Instance.Config.MemoryUsage == MemoryUsage.Optimal)
            {
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
                    var writeTask = File.WriteAllBytesAsync(frameFileName, pngData);
                    yield return new WaitUntil(() => writeTask.IsCompleted);
                    
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
        /// Generate a unique hash for this character based on name, gender, pitch, speed and image
        /// </summary>
        private string GenerateCharacterHash()
        {
            string nameHash = Name.GetHashCode().ToString("X8");
            string genderHash = Gender.ToString().GetHashCode().ToString("X8");
            string pitchHash = Pitch.ToString().GetHashCode().ToString("X8");
            string speedHash = Speed.ToString().GetHashCode().ToString("X8");
            string imageHash = Image != null ? TextureUtils.GenerateTextureHash(Image) : "00000000";

            // use mixin for hashing
            return StringUtils.MixHash(nameHash, genderHash, pitchHash, speedHash, imageHash);
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

        /// <summary>
        /// Generate voice sample using SparkTTS with character parameters
        /// </summary>
        private async Task GenerateVoiceSample(string voiceFolder)
        {
            // Convert enums to string parameters for SparkTTS
            string genderParam = ConvertGenderToString(Gender);
            string pitchParam = ConvertPitchToString(Pitch);
            string speedParam = ConvertSpeedToString(Speed);

            Logger.LogVerbose($"[Character] Generating voice sample with parameters: Gender={genderParam}, Pitch={pitchParam}, Speed={speedParam}");

            var characterVoice = await CharacterVoiceFactory.Instance.CreateFromStyleAsync(
                gender: genderParam,
                pitch: pitchParam,
                speed: speedParam,
                referenceText: Intro
            );

            if (characterVoice != null)
            {
                await characterVoice.SaveVoiceAsync(voiceFolder);
                characterVoice.Dispose();
            }
            else
            {
                Logger.LogError("[Character] Failed to create character voice");
            }
        }

        /// <summary>
        /// Convert Gender enum to SparkTTS string parameter
        /// </summary>
        private string ConvertGenderToString(Gender gender)
        {
            return gender switch
            {
                Gender.Male => "male",
                Gender.Female => "female",
                _ => "female"
            };
        }

        /// <summary>
        /// Convert Pitch enum to SparkTTS string parameter
        /// </summary>
        private string ConvertPitchToString(Pitch pitch)
        {
            return pitch switch
            {
                Pitch.VeryLow => "very_low",
                Pitch.Low => "low",
                Pitch.Moderate => "moderate",
                Pitch.High => "high",
                Pitch.VeryHigh => "very_high",
                _ => "moderate"
            };
        }

        /// <summary>
        /// Convert Speed enum to SparkTTS string parameter
        /// </summary>
        private string ConvertSpeedToString(Speed speed)
        {
            return speed switch
            {
                Speed.VeryLow => "very_low",
                Speed.Low => "low",
                Speed.Moderate => "moderate",
                Speed.High => "high",
                Speed.VeryHigh => "very_high",
                _ => "moderate"
            };
        }

        /// <summary>
        /// Load all expression data (frames, latents, face data)
        /// </summary>
        private IEnumerator LoadExpressionsData()
        {   
            string drivingFramesFolder = Path.Combine(CharacterFolder, "drivingFrames");
            if (!Directory.Exists(drivingFramesFolder))
            {
                Logger.LogWarning($"[CharacterFactory] No driving frames folder found: {drivingFramesFolder}");
                yield break;
            }

            var expressionFolders = Directory.GetDirectories(drivingFramesFolder);
            Logger.LogVerbose($"[CharacterFactory] Found {expressionFolders.Length} expression folders");

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
                    Logger.LogVerbose($"[CharacterFactory] Loaded expression {expressionIndex} ({expressionData.ExpressionName}): {expressionData.Data.FaceRegions.Count} frames");
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
                Logger.LogWarning($"[CharacterFactory] No latents file found: {latentsFile}");
                yield break;
            }

            var readTask = File.ReadAllBytesAsync(latentsFile);
            yield return new WaitUntil(() => readTask.IsCompleted);

            if (!readTask.IsFaulted)
            {
                var latentsBytes = readTask.Result;
                
                // Process latents in parallel using unsafe code for optimal performance
                var processTask = Task.Run(() => ProcessLatentsUnsafe(latentsBytes, expressionData));
                yield return new WaitUntil(() => processTask.IsCompleted);
                
                if (processTask.IsFaulted)
                {
                    Logger.LogError($"[CharacterFactory] Error processing latents: {processTask.Exception?.InnerException?.Message}");
                }
            }
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
                Logger.LogWarning("[CharacterFactory] No valid latents found in file");
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
                Logger.LogWarning($"[CharacterFactory] No faces file found: {facesFile}");
                yield break;
            }

            var readTask = File.ReadAllTextAsync(facesFile);
            yield return new WaitUntil(() => readTask.IsCompleted);

            if (!readTask.IsFaulted)
            {
                var facesJson = readTask.Result;
                var parseTask = ParseFaceDataJson(facesJson, expressionData, expressionFolder);
                yield return new WaitUntil(() => parseTask.IsCompleted);
            }
        }

        /// <summary>
        /// Load voice data for the character from the saved reference sample
        /// </summary>
        private IEnumerator LoadVoiceData()
        {
            string voiceFolder = Path.Combine(CharacterFolder, "voice");
            if (!Directory.Exists(voiceFolder))
            {
                Logger.LogWarning($"[CharacterFactory] No voice folder found: {voiceFolder}");
                yield break;
            }

            // Create character voice from the loaded reference sample
            var characterVoiceTask = CharacterVoiceFactory.Instance.CreateFromFolderAsync(voiceFolder);            
            yield return new WaitUntil(() => characterVoiceTask.IsCompleted);
            
            if (!characterVoiceTask.IsFaulted)
            {
                LoadedVoice = characterVoiceTask.Result;
                Logger.LogVerbose($"[CharacterFactory] Voice loaded from folder for {Name}");
            }
            else
            {
                Logger.LogError($"[CharacterFactory] Failed to create voice from folder: {characterVoiceTask.Exception?.Message}");
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
                Logger.LogError($"[CharacterFactory] Error parsing face data: {ex.Message}");
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
                Logger.LogError($"[CharacterFactory] Error loading face textures: {ex.Message}");
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
                    Logger.LogWarning($"[CharacterFactory] Texture file not found: {texturePath}");
                    return new Frame(); // Return empty frame
                }

                return new Frame(await File.ReadAllBytesAsync(texturePath), width, height);
            }
            catch (Exception ex)
            {
                Logger.LogError($"[CharacterFactory] Error loading texture {texturePath}: {ex.Message}");
                return new Frame();
            }
        }

        /// <summary>
        /// Load character data from the character folder or bundle
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
            var configJson = readConfigTask.Result;
            CharacterConfig config;
            try
            {
                config = JsonConvert.DeserializeObject<CharacterConfig>(configJson);
            }
            catch (Exception ex)
            {
                onError?.Invoke(new Exception($"Failed to parse character config: {ex.Message}"));
                yield break;
            }

            // Load character image
            string imagePath = Path.Combine(characterFolder, "image.png");
            if (!File.Exists(imagePath))
            {
                onError?.Invoke(new FileNotFoundException($"Character image not found: {imagePath}"));
                yield break;
            }

            var readImageTask = File.ReadAllBytesAsync(imagePath);
            yield return new WaitUntil(() => readImageTask.IsCompleted);

            if (readImageTask.IsFaulted)
            {
                onError?.Invoke(readImageTask.Exception?.InnerException ?? new Exception("Failed to read character image"));
                yield break;
            }

            // Create texture from image bytes
            var imageBytes = readImageTask.Result;
            var texture = new Texture2D(2, 2); // Temporary size, will be replaced by LoadImage
            if (!texture.LoadImage(imageBytes))
            {
                onError?.Invoke(new Exception("Failed to load character image into texture"));
                yield break;
            }

            // Create character object
            var character = new Character(
                config.name,
                config.gender,
                texture,
                config.pitch,
                config.speed,
                config.intro
            )
            {
                // Set character folder for data loading
                CharacterFolder = characterFolder
            };

            // Load all character data (expressions, voice, etc.)
            yield return character.LoadData();
            var elapsed = start.Elapsed;
            bool isBundle = characterFolder.EndsWith(".bundle");
            Logger.Log($"[Character] Character data for {character.Name} loaded from {(isBundle ? "bundle" : "folder")} in {elapsed.TotalMilliseconds} milliseconds");
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
    }

}
