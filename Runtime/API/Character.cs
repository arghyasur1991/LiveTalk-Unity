using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Video;
using SparkTTS;
using SparkTTS.Utils;
using Newtonsoft.Json;
using LiveTalk.Utils;

namespace LiveTalk.API
{
    public enum Gender
    {
        Male,
        Female
    }

    internal class ProcessFramesResult
    {
        public List<Texture2D> GeneratedFrames { get; set; } = new List<Texture2D>();
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

    public class Character
    {
        public string Name { get; internal set; }
        public Gender Gender { get; internal set; }
        public Texture2D Image { get; internal set; }
        public Pitch Pitch { get; internal set; }
        public Speed Speed { get; internal set; }
        public string Intro { get; internal set; }
        internal static string saveLocation;
        
        // Loaded character data for inference
        public bool IsDataLoaded { get; internal set; } = false;
        internal string CharacterFolder { get; set; }
        public Dictionary<int, ExpressionData> LoadedExpressions { get; internal set; } = new Dictionary<int, ExpressionData>();
        internal SparkTTS.CharacterVoice LoadedVoice { get; set; }
        
        /// <summary>
        /// Data for a specific expression including frames, latents, and face data
        /// </summary>
        public class ExpressionData
        {
            public List<float[]> Latents { get; set; } = new List<float[]>();
            internal List<Core.FaceData> FaceRegions { get; set; } = new List<Core.FaceData>();
            public string ExpressionName { get; set; }
            public int FaceRegionCount => FaceRegions?.Count ?? 0;
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

        public System.Collections.IEnumerator CreateAvatarAsync()
        {
            // Get the LiveTalkAPI instance
            var liveTalkAPI = CharacterFactory._liveTalkAPI;
            // Create folder for windows and package for mac in saveLocation

            // Adjust name to be a valid folder name
            // Generate Hash combining name, gender, and image hash
            
            // Organize folder as follows:
            // saveLocation/guid/
            // ---- drivingFrames/
            // -------- expression-0/
            // ---------- 00000.png // Frame 0
            // ---------- 00001.png // Frame 1
            // ---------- ...
            // ---------- latents.bin // Combined float array as binary file for all frames
            // ---------- faces.json // Combined face data for all frames as json
            // -------- expression-1/
            // ---------- 00000.png // Frame 0
            // ---------- 00001.png // Frame 1
            // ---------- ...
            // ---------- latents.bin
            // ---------- faces.json
            // -------- expression-2/
            // -------- ...
            // ---- voice/
            // -------- sample.wav // Generate audio from text sample using SparkTTS with gender, pitch, speed params.

            // Create avatar
            // 1. Create driving frames
            // 1.1. Create driving frames for each expression - talk_neutral, approve, disapprove, smile, sad, surprised, confused
            // 1.2. Create cacheable segment data and latents for each expression
            // 2. Create voice sample using SparkTTS with gender, pitch, speed params and intro text.
            if (liveTalkAPI == null)
                throw new InvalidOperationException("CharacterFactory not initialized. Call CharacterFactory.Initialize() first.");

            // Step 1: Generate a unique ID for this character based on name, gender, and image
            string characterId = GenerateCharacterHash();
            string characterFolder = Path.Combine(saveLocation, characterId);
            
            // Create main character folder (clean slate approach)
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

            // Save image (convert to uncompressed format if needed)
            string imagePath = Path.Combine(characterFolder, "image.png");
            var uncompressedImage = LiveTalk.Utils.TextureUtils.ConvertToUncompressedTexture(Image);
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

            Debug.Log($"[Character] Creating avatar for {Name} in folder: {characterFolder}");

            // Step 2: Generate driving frames for each expression
            var expressions = new string[] { "talk-neutral", "approve", "disapprove", "smile", "sad", "surprised", "confused" };
            
            for (int expressionIndex = 0; expressionIndex < 1 /*expressions.Length*/; expressionIndex++)
            {
                string expression = expressions[expressionIndex];
                string expressionFolder = Path.Combine(drivingFramesFolder, $"expression-{expressionIndex}");
                Directory.CreateDirectory(expressionFolder);

                Debug.Log($"[Character] Processing expression: {expression} (index: {expressionIndex})");

                // Load the driving video for this expression
                VideoClip drivingVideo = LoadDrivingVideoForExpression(expression);
                if (drivingVideo == null)
                {
                    Debug.LogWarning($"[Character] Could not load driving video for expression: {expression}");
                    continue;
                }

                // Process this expression with coroutines outside try-catch
                yield return ProcessExpressionCoroutine(expression, expressionIndex, drivingVideo, expressionFolder, liveTalkAPI);
            }

            // Step 3: Generate voice sample using SparkTTS
            var voiceTask = GenerateVoiceSample(voiceFolder);
            yield return new WaitUntil(() => voiceTask.IsCompleted);

            Debug.Log($"[Character] Avatar creation completed for {Name}");
        }

        /// <summary>
        /// Generate speech using the character's saved data and voice
        /// </summary>
        /// <param name="text">Text to speak</param>
        /// <param name="expressionIndex">Expression to use (0-6: talk-neutral, approve, disapprove, smile, sad, surprised, confused)</param>
        /// <returns>OutputStream for generated talking head frames</returns>
        public OutputStream Speak(string text, int expressionIndex = 0)
        {
            if (!IsDataLoaded)
            {
                throw new InvalidOperationException("Character data not loaded. Use CharacterFactory.LoadCharacterAsync() first.");
            }

            if (string.IsNullOrEmpty(text))
            {
                throw new ArgumentException("Text cannot be null or empty.");
            }

            if (!LoadedExpressions.ContainsKey(expressionIndex))
            {
                throw new ArgumentException($"Expression index {expressionIndex} not available. Available expressions: {string.Join(", ", LoadedExpressions.Keys)}");
            }

            if (LoadedVoice == null)
            {
                throw new InvalidOperationException("Character voice not loaded.");
            }

            var liveTalkAPI = CharacterFactory._liveTalkAPI;
            if (liveTalkAPI == null)
            {
                throw new InvalidOperationException("CharacterFactory not initialized. Call CharacterFactory.Initialize() first.");
            }

            Debug.Log($"[Character] {Name} speaking: \"{text}\" with expression {expressionIndex}");

            // Generate audio using the loaded character voice
            var audioClip = LoadedVoice.GenerateSpeechAsync(text).Result;
            if (audioClip == null)
            {
                throw new InvalidOperationException("Failed to generate speech audio.");
            }

            // Use the preloaded expression data for MuseTalk
            var expressionData = LoadedExpressions[expressionIndex];
            
            // Generate talking head using MuseTalk with preloaded data
            return liveTalkAPI.GenerateTalkingHeadWithPreloadedData(
                expressionData.Latents,
                expressionData.FaceRegions,
                audioClip
            );
        }

        /// <summary>
        /// Generate speech asynchronously using coroutines
        /// </summary>
        /// <param name="text">Text to speak</param>
        /// <param name="expressionIndex">Expression to use</param>
        /// <param name="onComplete">Callback when audio generation is complete</param>
        /// <param name="onError">Callback when an error occurs</param>
        /// <returns>Coroutine for audio generation, then OutputStream for video frames</returns>
        public System.Collections.IEnumerator SpeakAsync(
            string text, 
            int expressionIndex = 0,
            System.Action<OutputStream> onComplete = null,
            System.Action<System.Exception> onError = null)
        {
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

            if (!LoadedExpressions.ContainsKey(expressionIndex))
            {
                onError?.Invoke(new ArgumentException($"Expression index {expressionIndex} not available. Available expressions: {string.Join(", ", LoadedExpressions.Keys)}"));
                yield break;
            }

            if (LoadedVoice == null)
            {
                onError?.Invoke(new InvalidOperationException("Character voice not loaded."));
                yield break;
            }

            var liveTalkAPI = CharacterFactory._liveTalkAPI;
            if (liveTalkAPI == null)
            {
                onError?.Invoke(new InvalidOperationException("CharacterFactory not initialized. Call CharacterFactory.Initialize() first."));
                yield break;
            }

            Debug.Log($"[Character] {Name} speaking async: \"{text}\" with expression {expressionIndex}");

            // Generate audio using the loaded character voice
            var audioTask = LoadedVoice.GenerateSpeechAsync(text);
            yield return new UnityEngine.WaitUntil(() => audioTask.IsCompleted);

            if (audioTask.IsFaulted)
            {
                onError?.Invoke(audioTask.Exception?.InnerException ?? new System.Exception("Failed to generate speech audio."));
                yield break;
            }

            var audioClip = audioTask.Result;
            if (audioClip == null)
            {
                onError?.Invoke(new InvalidOperationException("Generated audio clip is null."));
                yield break;
            }

            // Use the preloaded expression data for MuseTalk
            var expressionData = LoadedExpressions[expressionIndex];
            
            // Generate talking head using MuseTalk with preloaded data
            var outputStream = liveTalkAPI.GenerateTalkingHeadWithPreloadedData(
                expressionData.Latents,
                expressionData.FaceRegions,
                audioClip
            );

            onComplete?.Invoke(outputStream);
        }

        /// <summary>
        /// Process a single expression with coroutines to handle frame streaming
        /// </summary>
        private System.Collections.IEnumerator ProcessExpressionCoroutine(
            string expression, 
            int expressionIndex, 
            VideoClip drivingVideo, 
            string expressionFolder, 
            LiveTalkAPI liveTalkAPI)
        {
            // Create a temporary VideoPlayer to process the driving frames
            var tempGO = new GameObject("TempVideoPlayer");
            var videoPlayer = tempGO.AddComponent<VideoPlayer>();
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

            Debug.Log($"[Character] Generated and saved {processResult.GeneratedFrames.Count} frames for expression: {expression}");

            // Generate and save cache data
            var cacheTask = GenerateAndSaveCacheData(expressionFolder, processResult.GeneratedFrames);
            yield return new WaitUntil(() => cacheTask.IsCompleted);

            // Clean up generated textures
            foreach (var frame in processResult.GeneratedFrames)
            {
                UnityEngine.Object.DestroyImmediate(frame);
            }

            // Clean up temp GameObject
            UnityEngine.Object.DestroyImmediate(tempGO);
        }

        /// <summary>
        /// Process frame stream using coroutines
        /// </summary>
        private System.Collections.IEnumerator ProcessFramesCoroutine(OutputStream outputStream, string expressionFolder, ProcessFramesResult result)
        {
            int frameIndex = 0;
            
            // Process frames as they become available using coroutine pattern
            while (!outputStream.Finished)
            {
                var awaiter = outputStream.WaitForNext();
                yield return awaiter;
                
                if (awaiter.Texture != null)
                {
                    // Save LivePortrait generated frames as numbered PNGs (these are the driving frames)
                    string frameFileName = Path.Combine(expressionFolder, $"{frameIndex:D5}.png");
                    byte[] pngData = awaiter.Texture.EncodeToPNG();
                    var writeTask = File.WriteAllBytesAsync(frameFileName, pngData);
                    yield return new UnityEngine.WaitUntil(() => writeTask.IsCompleted);
                    
                    // Keep reference for cache generation
                    result.GeneratedFrames.Add(awaiter.Texture);
                    frameIndex++;
                }
            }
        }

        /// <summary>
        /// Generate a unique hash for this character based on name, gender, and image
        /// </summary>
        private string GenerateCharacterHash()
        {
            string nameHash = Name?.GetHashCode().ToString("X8") ?? "00000000";
            string genderHash = Gender.ToString().GetHashCode().ToString("X8");
            string imageHash = Image != null ? LiveTalk.Utils.TextureUtils.GenerateTextureHash(Image) : "00000000";
            
            return $"{nameHash}_{genderHash}_{imageHash}";
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
                    Debug.Log($"[Character] Loaded driving video: {path}");
                    return videoClip;
                }
            }

            Debug.LogWarning($"[Character] Could not find driving video for expression: {expression}");
            return null;
        }

        /// <summary>
        /// Generate and save cache data (latents and face data) for the processed frames using real MuseTalkInference
        /// </summary>
        private async Task GenerateAndSaveCacheData(string expressionFolder, List<Texture2D> frames)
        {
            try
            {
                Debug.Log($"[Character] Generating real cache data for {frames.Count} frames using MuseTalkInference");

                // Create a temporary MuseTalkInference instance for processing
                var liveTalkAPI = CharacterFactory._liveTalkAPI;
                if (liveTalkAPI == null)
                {
                    Debug.LogError("[Character] LiveTalkAPI not available for cache generation");
                    return;
                }

                // Convert frames to avatar textures array for processing
                var avatarTextures = frames.ToArray();

                // Use MuseTalkInference to process the avatar images and extract real data
                var avatarData = await ProcessAvatarImagesWithMuseTalk(liveTalkAPI, avatarTextures);

                if (avatarData != null && avatarData.Latents.Count > 0)
                {
                    // Save real latents data
                    await SaveLatentsToFile(expressionFolder, avatarData.Latents);

                    // Save real face data
                    await SaveFaceDataToFile(expressionFolder, avatarData.FaceRegions);

                    Debug.Log($"[Character] Generated real cache data: {avatarData.Latents.Count} latents, {avatarData.FaceRegions.Count} face regions");
                }
                else
                {
                    throw new InvalidOperationException("Failed to generate avatar data using real MuseTalk processing. No fallback available.");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Character] Error generating real cache data: {ex.Message}");
                throw new InvalidOperationException($"Failed to generate real cache data: {ex.Message}. No fallback available.", ex);
            }
        }

        /// <summary>
        /// Process avatar images using MuseTalkInference public API to extract real latents and face data
        /// This uses the actual MuseTalk face analysis and VAE encoder pipeline - NO FALLBACKS
        /// </summary>
        private async Task<LiveTalk.Core.AvatarData> ProcessAvatarImagesWithMuseTalk(LiveTalkAPI liveTalkAPI, Texture2D[] avatarTextures)
        {
            Debug.Log($"[Character] Processing {avatarTextures.Length} avatar textures using real MuseTalk pipeline");

            // Use the public MuseTalk ProcessAvatarImages API directly - no reflection needed
            var avatarData = await liveTalkAPI.MuseTalk.ProcessAvatarImages(avatarTextures);
            
            if (avatarData?.FaceRegions?.Count == 0 || avatarData?.Latents?.Count == 0)
            {
                throw new InvalidOperationException($"Real MuseTalk processing failed to generate valid avatar data. FaceRegions: {avatarData?.FaceRegions?.Count ?? 0}, Latents: {avatarData?.Latents?.Count ?? 0}");
            }

            Debug.Log($"[Character] Real MuseTalk processing completed: {avatarData.Latents.Count} latents, {avatarData.FaceRegions.Count} face regions");
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
                
                Debug.Log($"[Character] Saved {latents.Count} latent arrays ({totalFloats} total floats) to {latentsFile}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Character] Error saving latents: {ex.Message}");
            }
        }

        /// <summary>
        /// Save real face data to JSON file and save all precomputed textures
        /// </summary>
        private async Task SaveFaceDataToFile(string expressionFolder, List<LiveTalk.Core.FaceData> faceRegions)
        {
            try
            {
                var facesFile = Path.Combine(expressionFolder, "faces.json");
                var texturesFolder = Path.Combine(expressionFolder, "textures");
                
                // Create texture subfolders (removed "original" to eliminate redundancy)
                var subfolders = new[]
                {
                    "cropped", "faceLarge", "segmentationMask",
                    "maskSmall", "fullMask", "boundaryMask", "blurredMask"
                };
                
                foreach (var subfolder in subfolders)
                {
                    Directory.CreateDirectory(Path.Combine(texturesFolder, subfolder));
                }

                Debug.Log($"[Character] Saving face data with precomputed textures for {faceRegions.Count} face regions");

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
                
                string json = Newtonsoft.Json.JsonConvert.SerializeObject(faceDataJson, Newtonsoft.Json.Formatting.Indented);
                await File.WriteAllTextAsync(facesFile, json);
                
                Debug.Log($"[Character] Saved complete face data with textures for {faceRegions.Count} face regions to {facesFile}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Character] Error saving face data with textures: {ex.Message}");
            }
        }

        /// <summary>
        /// Save all precomputed textures for a single face region
        /// </summary>
        private async Task<Dictionary<string, string>> SaveFaceTextures(string texturesFolder, LiveTalk.Core.FaceData face, int faceIndex)
        {
            var texturePaths = new Dictionary<string, string>();
            
            try
            {
                // Define texture mappings: texture data -> folder name -> filename
                // Note: Removed "original" to eliminate redundancy - driving frames are saved as numbered PNGs
                var textureMap = new List<(LiveTalk.Core.Frame frame, string folder, string key)>
                {
                    (face.CroppedFaceTexture, "cropped", "croppedFace"),
                    (face.FaceLarge, "faceLarge", "faceLarge"),
                    (face.SegmentationMask, "segmentationMask", "segmentationMask"),
                    (face.MaskSmall, "maskSmall", "maskSmall"),
                    (face.FullMask, "fullMask", "fullMask"),
                    (face.BoundaryMask, "boundaryMask", "boundaryMask"),
                    (face.BlurredMask, "blurredMask", "blurredMask")
                };

                foreach (var (frame, folder, key) in textureMap)
                {
                    if (frame.data != null && frame.data.Length > 0)
                    {
                        string filename = $"face_{faceIndex:D3}.png";
                        string folderPath = Path.Combine(texturesFolder, folder);
                        string fullPath = Path.Combine(folderPath, filename);
                        
                        // Convert Frame to Texture2D and save as PNG
                        var texture = LiveTalk.Utils.TextureUtils.FrameToTexture2D(frame);
                        byte[] pngData = texture.EncodeToPNG();
                        await File.WriteAllBytesAsync(fullPath, pngData);
                        
                        // Store relative path for JSON reference
                        string relativePath = Path.Combine("textures", folder, filename).Replace('\\', '/');
                        texturePaths[key] = relativePath;
                        
                        // Clean up texture
                        UnityEngine.Object.DestroyImmediate(texture);
                        
                        Debug.Log($"[Character] Saved {key} texture: {relativePath} ({frame.width}x{frame.height})");
                    }
                    else
                    {
                        texturePaths[key] = null; // Mark as missing/empty
                        Debug.LogWarning($"[Character] {key} texture data is null or empty for face {faceIndex}");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Character] Error saving textures for face {faceIndex}: {ex.Message}");
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

            Debug.Log($"[Character] Generating voice sample with parameters: Gender={genderParam}, Pitch={pitchParam}, Speed={speedParam}");

            var characterVoice = await CharacterFactory._characterVoiceFactory.CreateFromStyleAsync(
                gender: genderParam,
                pitch: pitchParam,
                speed: speedParam,
                referenceText: Intro ?? "Hello, I am a character."
            );

            if (characterVoice != null)
            {
                // Generate the voice sample
                var audioClip = characterVoice.ReferenceClip;

                if (audioClip != null)
                {
                    // Convert AudioClip to WAV and save
                    string samplePath = Path.Combine(voiceFolder, "sample.wav");
                    await SaveAudioClipAsWAV(audioClip, samplePath);
                    Debug.Log($"[Character] Voice sample saved to: {samplePath}");

                    // Also save voice config for reference
                    var voiceConfig = new
                    {
                        gender = genderParam,
                        pitch = pitchParam,
                        speed = speedParam,
                        introText = Intro ?? "Hello, I am a character in this mystery.",
                        timestamp = DateTime.UtcNow,
                        audioFile = "sample.wav",
                        sampleRate = audioClip.frequency,
                        channels = audioClip.channels,
                        length = audioClip.length
                    };
                    
                    string configPath = Path.Combine(voiceFolder, "voice_config.json");
                    string configJson = Newtonsoft.Json.JsonConvert.SerializeObject(voiceConfig, Newtonsoft.Json.Formatting.Indented);
                    await File.WriteAllTextAsync(configPath, configJson);
                }
                else
                {
                    Debug.LogError("[Character] Failed to generate voice sample audio");
                }

                // Clean up
                characterVoice.Dispose();
            }
            else
            {
                Debug.LogError("[Character] Failed to create character voice");
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
        /// Save AudioClip as WAV file using SparkTTS AudioLoaderService
        /// </summary>
        private async Task SaveAudioClipAsWAV(AudioClip audioClip, string filePath)
        {
            try
            {
                // Use SparkTTS AudioLoaderService to save the audio clip directly
                await AudioLoaderService.SaveAudioClipToFile(audioClip, filePath);
                Debug.Log($"[Character] Audio saved as WAV: {filePath} ({audioClip.samples} samples, {audioClip.frequency}Hz, {audioClip.channels} channels)");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Character] Error saving audio as WAV: {ex.Message}");
            }
        }
    }

    public static class CharacterFactory
    {
        internal static LiveTalkAPI _liveTalkAPI = null;
        internal static CharacterVoiceFactory _characterVoiceFactory = null;
        private static bool _initialized = false;

        public static void Initialize(
            string saveLocation,
            LiveTalkController avatarController)
        {
            if (_liveTalkAPI == null && !_initialized)
            {
                _liveTalkAPI = LivePortraitMuseTalkFactory.Create(avatarController);
                Character.saveLocation = saveLocation;
                _characterVoiceFactory = new CharacterVoiceFactory();
                _initialized = true;
            }
        }

        public static System.Collections.IEnumerator CreateCharacterAsync(
            string name,
            Gender gender,
            Texture2D image,
            Pitch pitch,
            Speed speed,
            string intro,
            System.Action<Character> onComplete,
            System.Action<System.Exception> onError)
        {
            if (!_initialized)
            {
                onError?.Invoke(new System.Exception("CharacterFactory not initialized. Call Initialize() first."));
                yield break;
            }

            var character = new Character(name, gender, image, pitch, speed, intro);
            yield return character.CreateAvatarAsync();
            onComplete?.Invoke(character);
        }

        /// <summary>
        /// Load a character from the saveLocation using the character GUID
        /// </summary>
        /// <param name="characterId">The GUID/hash of the character to load</param>
        /// <param name="onComplete">Callback when character is successfully loaded</param>
        /// <param name="onError">Callback when an error occurs</param>
        public static System.Collections.IEnumerator LoadCharacterAsync(
            string characterId,
            System.Action<Character> onComplete,
            System.Action<System.Exception> onError)
        {
            if (!_initialized)
            {
                onError?.Invoke(new System.Exception("CharacterFactory not initialized. Call Initialize() first."));
                yield break;
            }

            if (string.IsNullOrEmpty(characterId))
            {
                onError?.Invoke(new System.ArgumentException("Character ID cannot be null or empty."));
                yield break;
            }

            string characterFolder = Path.Combine(Character.saveLocation, characterId);
            if (!Directory.Exists(characterFolder))
            {
                onError?.Invoke(new System.IO.DirectoryNotFoundException($"Character folder not found: {characterFolder}"));
                yield break;
            }

            Character loadedCharacter = null;
            System.Exception loadError = null;

            // Load character data in a coroutine
            yield return LoadCharacterDataCoroutine(characterFolder, 
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
                onError?.Invoke(new System.Exception("Failed to load character: Unknown error"));
            }
        }

        /// <summary>
        /// Get all available character IDs from the saveLocation
        /// </summary>
        /// <returns>Array of character GUIDs/hashes</returns>
        public static string[] GetAvailableCharacterIds()
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
                        characterIds.Add(dirName);
                    }
                }

                return characterIds.ToArray();
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[CharacterFactory] Error getting available character IDs: {ex.Message}");
                return new string[0];
            }
        }

        /// <summary>
        /// Load character data from the character folder
        /// </summary>
        private static System.Collections.IEnumerator LoadCharacterDataCoroutine(
            string characterFolder,
            System.Action<Character> onComplete,
            System.Action<System.Exception> onError)
        {
            // Load character.json
            string configPath = Path.Combine(characterFolder, "character.json");
            if (!File.Exists(configPath))
            {
                onError?.Invoke(new System.IO.FileNotFoundException($"Character config file not found: {configPath}"));
                yield break;
            }

            var readConfigTask = File.ReadAllTextAsync(configPath);
            yield return new UnityEngine.WaitUntil(() => readConfigTask.IsCompleted);

            if (readConfigTask.IsFaulted)
            {
                onError?.Invoke(readConfigTask.Exception?.InnerException ?? new System.Exception("Failed to read character config"));
                yield break;
            }

            // Parse character config
            var configJson = readConfigTask.Result;
            CharacterConfig config;
            try
            {
                config = JsonConvert.DeserializeObject<CharacterConfig>(configJson);
            }
            catch (System.Exception ex)
            {
                onError?.Invoke(new System.Exception($"Failed to parse character config: {ex.Message}"));
                yield break;
            }

            // Load character image
            string imagePath = Path.Combine(characterFolder, "image.png");
            if (!File.Exists(imagePath))
            {
                onError?.Invoke(new System.IO.FileNotFoundException($"Character image not found: {imagePath}"));
                yield break;
            }

            var readImageTask = File.ReadAllBytesAsync(imagePath);
            yield return new UnityEngine.WaitUntil(() => readImageTask.IsCompleted);

            if (readImageTask.IsFaulted)
            {
                onError?.Invoke(readImageTask.Exception?.InnerException ?? new System.Exception("Failed to read character image"));
                yield break;
            }

            // Create texture from image bytes
            var imageBytes = readImageTask.Result;
            var texture = new Texture2D(2, 2); // Temporary size, will be replaced by LoadImage
            if (!texture.LoadImage(imageBytes))
            {
                onError?.Invoke(new System.Exception("Failed to load character image into texture"));
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
            );

            // Set character folder for data loading
            character.CharacterFolder = characterFolder;

            // Load all character data (expressions, voice, etc.)
            yield return LoadCharacterDataContents(character);

            onComplete?.Invoke(character);
        }

        /// <summary>
        /// Load all character data including expressions, voice, and precomputed data
        /// </summary>
        private static System.Collections.IEnumerator LoadCharacterDataContents(Character character)
        {
            Debug.Log($"[CharacterFactory] Loading character data for {character.Name}");

            // Load expressions data
            yield return LoadExpressionsData(character);

            // Load voice data
            yield return LoadVoiceData(character);

            character.IsDataLoaded = true;
            Debug.Log($"[CharacterFactory] Character data loaded successfully for {character.Name}");
        }

        /// <summary>
        /// Load all expression data (frames, latents, face data)
        /// </summary>
        private static System.Collections.IEnumerator LoadExpressionsData(Character character)
        {
            string drivingFramesFolder = System.IO.Path.Combine(character.CharacterFolder, "drivingFrames");
            if (!System.IO.Directory.Exists(drivingFramesFolder))
            {
                Debug.LogWarning($"[CharacterFactory] No driving frames folder found: {drivingFramesFolder}");
                yield break;
            }

            var expressionFolders = System.IO.Directory.GetDirectories(drivingFramesFolder);
            Debug.Log($"[CharacterFactory] Found {expressionFolders.Length} expression folders");

            for (int i = 0; i < expressionFolders.Length; i++)
            {
                string expressionFolder = expressionFolders[i];
                string folderName = System.IO.Path.GetFileName(expressionFolder);
                
                // Extract expression index from folder name (expression-0, expression-1, etc.)
                if (folderName.StartsWith("expression-") && int.TryParse(folderName.Substring(11), out int expressionIndex))
                {
                    var expressionData = new Character.ExpressionData();
                    expressionData.ExpressionName = GetExpressionName(expressionIndex);

                    // Load latents
                    yield return LoadExpressionLatents(expressionFolder, expressionData);

                    // Load face data
                    yield return LoadExpressionFaceData(expressionFolder, expressionData);

                    // Load frames
                    yield return LoadExpressionFrames(expressionFolder, expressionData);

                    character.LoadedExpressions[expressionIndex] = expressionData;
                    Debug.Log($"[CharacterFactory] Loaded expression {expressionIndex} ({expressionData.ExpressionName}): {expressionData.Latents.Count} latents, {expressionData.FaceRegions.Count} face regions");
                }
            }
        }

        /// <summary>
        /// Load frames for a specific expression from numbered PNGs
        /// </summary>
        private static System.Collections.IEnumerator LoadExpressionFrames(string expressionFolder, Character.ExpressionData expressionData)
        {
            var frameFiles = System.IO.Directory.GetFiles(expressionFolder, "*.png")
                .Where(f => System.IO.Path.GetFileName(f).StartsWith("00"))
                .OrderBy(f => f)
                .ToArray();

            for (int i = 0; i < frameFiles.Length; i++)
            {
                string frameFile = frameFiles[i];
                var readTask = System.IO.File.ReadAllBytesAsync(frameFile);
                yield return new UnityEngine.WaitUntil(() => readTask.IsCompleted);

                if (!readTask.IsFaulted)
                {
                    var frameBytes = readTask.Result;
                    var texture = new Texture2D(2, 2);
                    if (texture.LoadImage(frameBytes))
                    {
                        expressionData.FaceRegions[i].OriginalTexture = 
                                TextureUtils.Texture2DToFrame(TextureUtils.ConvertTexture2DToRGB24(texture));
                    }
                }
            }
        }

        /// <summary>
        /// Load latents for a specific expression
        /// </summary>
        private static System.Collections.IEnumerator LoadExpressionLatents(string expressionFolder, Character.ExpressionData expressionData)
        {
            string latentsFile = System.IO.Path.Combine(expressionFolder, "latents.bin");
            if (!System.IO.File.Exists(latentsFile))
            {
                Debug.LogWarning($"[CharacterFactory] No latents file found: {latentsFile}");
                yield break;
            }

            var readTask = System.IO.File.ReadAllBytesAsync(latentsFile);
            yield return new UnityEngine.WaitUntil(() => readTask.IsCompleted);

            if (!readTask.IsFaulted)
            {
                var latentsBytes = readTask.Result;
                
                // Convert bytes back to float array
                var allLatents = new float[latentsBytes.Length / sizeof(float)];
                System.Buffer.BlockCopy(latentsBytes, 0, allLatents, 0, latentsBytes.Length);

                // Split back into individual latent arrays (assuming each latent is 8*32*32 = 8192 floats)
                const int latentSize = 8 * 32 * 32; // 8192 floats per latent
                int numLatents = allLatents.Length / latentSize;

                for (int i = 0; i < numLatents; i++)
                {
                    var latent = new float[latentSize];
                    System.Array.Copy(allLatents, i * latentSize, latent, 0, latentSize);
                    expressionData.Latents.Add(latent);
                }
            }
        }

        /// <summary>
        /// Load face data for a specific expression
        /// </summary>
        private static System.Collections.IEnumerator LoadExpressionFaceData(string expressionFolder, Character.ExpressionData expressionData)
        {
            string facesFile = System.IO.Path.Combine(expressionFolder, "faces.json");
            if (!System.IO.File.Exists(facesFile))
            {
                Debug.LogWarning($"[CharacterFactory] No faces file found: {facesFile}");
                yield break;
            }

            var readTask = System.IO.File.ReadAllTextAsync(facesFile);
            yield return new UnityEngine.WaitUntil(() => readTask.IsCompleted);

            if (!readTask.IsFaulted)
            {
                var facesJson = readTask.Result;
                ParseFaceDataJson(facesJson, expressionData, expressionFolder);
            }
        }

        /// <summary>
        /// Load voice data for the character from the saved reference sample
        /// </summary>
        private static System.Collections.IEnumerator LoadVoiceData(Character character)
        {
            string voiceFolder = System.IO.Path.Combine(character.CharacterFolder, "voice");
            string voiceSampleFile = System.IO.Path.Combine(voiceFolder, "sample.wav");
            string voiceConfigFile = System.IO.Path.Combine(voiceFolder, "voice_config.json");
            
            if (!System.IO.File.Exists(voiceSampleFile))
            {
                Debug.LogWarning($"[CharacterFactory] No voice sample found: {voiceSampleFile}");
                yield break;
            }

            if (!System.IO.File.Exists(voiceConfigFile))
            {
                Debug.LogWarning($"[CharacterFactory] No voice config found: {voiceConfigFile}");
                yield break;
            }

            // Read the voice config for metadata
            var readConfigTask = System.IO.File.ReadAllTextAsync(voiceConfigFile);
            yield return new UnityEngine.WaitUntil(() => readConfigTask.IsCompleted);

            if (readConfigTask.IsFaulted)
            {
                Debug.LogError($"[CharacterFactory] Failed to read voice config: {readConfigTask.Exception?.Message}");
                yield break;
            }

                        // Load the reference audio sample using AudioLoaderService
            var loadAudioTask = AudioLoaderService.LoadAudioClipAsync(voiceSampleFile);
            yield return new UnityEngine.WaitUntil(() => loadAudioTask.IsCompleted);
            
            if (loadAudioTask.IsFaulted)
            {
                Debug.LogError($"[CharacterFactory] Failed to load audio sample: {loadAudioTask.Exception?.Message}");
                yield break;
            }
            
            var audioClip = loadAudioTask.Result;
            if (audioClip != null)
            {
                // Create character voice from the loaded reference sample
                var characterVoiceTask = _characterVoiceFactory.CreateFromReferenceAsync(audioClip);
                
                yield return new UnityEngine.WaitUntil(() => characterVoiceTask.IsCompleted);
                
                if (!characterVoiceTask.IsFaulted)
                {
                    character.LoadedVoice = characterVoiceTask.Result;
                    Debug.Log($"[CharacterFactory] Voice loaded from reference sample for {character.Name}");
                }
                else
                {
                    Debug.LogError($"[CharacterFactory] Failed to create voice from reference: {characterVoiceTask.Exception?.Message}");
                }
            }
            else
            {
                Debug.LogError($"[CharacterFactory] Failed to load audio clip from: {voiceSampleFile}");
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
        private static void ParseFaceDataJson(string facesJson, Character.ExpressionData expressionData, string expressionFolder)
        {
            try
            {
                // Parse the JSON using a proper data structure instead of dynamic
                var faceDataJson = JsonConvert.DeserializeObject<FaceDataContainer>(facesJson);
                
                if (faceDataJson?.faceRegions != null)
                {
                    foreach (var faceRegion in faceDataJson.faceRegions)
                    {
                        // Create complete face data structure with all loaded textures
                        var faceData = new LiveTalk.Core.FaceData
                        {
                            HasFace = faceRegion.hasFace,
                            BoundingBox = new Rect(
                                faceRegion.boundingBox.x,
                                faceRegion.boundingBox.y,
                                faceRegion.boundingBox.width,
                                faceRegion.boundingBox.height
                            ),
                            AdjustedFaceBbox = new UnityEngine.Vector4(
                                faceRegion.adjustedFaceBbox.x,
                                faceRegion.adjustedFaceBbox.y,
                                faceRegion.adjustedFaceBbox.z,
                                faceRegion.adjustedFaceBbox.w
                            ),
                            CropBox = new UnityEngine.Vector4(
                                faceRegion.cropBox.x,
                                faceRegion.cropBox.y,
                                faceRegion.cropBox.z,
                                faceRegion.cropBox.w
                            )
                        };

                        // Load all saved textures from the texture files
                        LoadFaceTextures(faceData, faceRegion, expressionFolder);

                        expressionData.FaceRegions.Add(faceData);
                    }
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[CharacterFactory] Error parsing face data: {ex.Message}");
            }
        }

        /// <summary>
        /// Load all face textures from saved files
        /// </summary>
        private static void LoadFaceTextures(LiveTalk.Core.FaceData faceData, FaceRegionData faceRegion, string expressionFolder)
        {
            try
            {
                // Load cropped face texture
                if (!string.IsNullOrEmpty(faceRegion.textureFiles?.croppedFace))
                {
                    string croppedPath = System.IO.Path.Combine(expressionFolder, faceRegion.textureFiles.croppedFace);
                    faceData.CroppedFaceTexture = LoadTextureAsFrame(croppedPath);
                }

                // Load face large texture
                if (!string.IsNullOrEmpty(faceRegion.textureFiles?.faceLarge))
                {
                    string faceLargePath = System.IO.Path.Combine(expressionFolder, faceRegion.textureFiles.faceLarge);
                    faceData.FaceLarge = LoadTextureAsFrame(faceLargePath);
                }

                // Load segmentation mask
                if (!string.IsNullOrEmpty(faceRegion.textureFiles?.segmentationMask))
                {
                    string segmentationPath = System.IO.Path.Combine(expressionFolder, faceRegion.textureFiles.segmentationMask);
                    faceData.SegmentationMask = LoadTextureAsFrame(segmentationPath);
                }

                // Load mask small
                if (!string.IsNullOrEmpty(faceRegion.textureFiles?.maskSmall))
                {
                    string maskSmallPath = System.IO.Path.Combine(expressionFolder, faceRegion.textureFiles.maskSmall);
                    faceData.MaskSmall = LoadTextureAsFrame(maskSmallPath);
                }

                // Load full mask
                if (!string.IsNullOrEmpty(faceRegion.textureFiles?.fullMask))
                {
                    string fullMaskPath = System.IO.Path.Combine(expressionFolder, faceRegion.textureFiles.fullMask);
                    faceData.FullMask = LoadTextureAsFrame(fullMaskPath);
                }

                // Load boundary mask
                if (!string.IsNullOrEmpty(faceRegion.textureFiles?.boundaryMask))
                {
                    string boundaryMaskPath = System.IO.Path.Combine(expressionFolder, faceRegion.textureFiles.boundaryMask);
                    faceData.BoundaryMask = LoadTextureAsFrame(boundaryMaskPath);
                }

                // Load blurred mask
                if (!string.IsNullOrEmpty(faceRegion.textureFiles?.blurredMask))
                {
                    string blurredMaskPath = System.IO.Path.Combine(expressionFolder, faceRegion.textureFiles.blurredMask);
                    faceData.BlurredMask = LoadTextureAsFrame(blurredMaskPath);
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[CharacterFactory] Error loading face textures: {ex.Message}");
            }
        }

        /// <summary>
        /// Load a texture file and convert it to Frame format
        /// </summary>
        private static LiveTalk.Core.Frame LoadTextureAsFrame(string texturePath)
        {
            try
            {
                if (!System.IO.File.Exists(texturePath))
                {
                    Debug.LogWarning($"[CharacterFactory] Texture file not found: {texturePath}");
                    return new LiveTalk.Core.Frame(); // Return empty frame
                }

                byte[] textureBytes = System.IO.File.ReadAllBytes(texturePath);
                var texture = new Texture2D(2, 2);
                
                if (texture.LoadImage(textureBytes))
                {
                    // Convert texture to Frame format
                    var frame = TextureUtils.Texture2DToFrame(TextureUtils.ConvertTexture2DToRGB24(texture));
                    UnityEngine.Object.DestroyImmediate(texture); // Clean up temporary texture
                    return frame;
                }
                else
                {
                    Debug.LogError($"[CharacterFactory] Failed to load image from: {texturePath}");
                    UnityEngine.Object.DestroyImmediate(texture);
                    return new LiveTalk.Core.Frame();
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[CharacterFactory] Error loading texture {texturePath}: {ex.Message}");
                return new LiveTalk.Core.Frame();
            }
        }


    }

    /// <summary>
    /// Configuration data structure for loading characters
    /// </summary>
    [System.Serializable]
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
    [System.Serializable]
    internal class FaceDataContainer
    {
        public FaceRegionData[] faceRegions;
    }

    [System.Serializable]
    internal class FaceRegionData
    {
        public bool hasFace;
        public BoundingBoxData boundingBox;
        public BoundingBoxData adjustedFaceBbox;
        public BoundingBoxData cropBox;
        public TextureFilesData textureFiles;
    }

    [System.Serializable]
    internal class BoundingBoxData
    {
        public float x;
        public float y;
        public float width;
        public float height;
        public float z; // For Vector4 data
        public float w; // For Vector4 data
    }

    [System.Serializable]
    internal class TextureFilesData
    {
        public string croppedFace;
        public string faceLarge;
        public string segmentationMask;
        public string maskSmall;
        public string fullMask;
        public string boundaryMask;
        public string blurredMask;
    }
}