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

namespace LiveTalk.API
{
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

    public class Character
    {
        public string Name { get; internal set; }
        public Gender Gender { get; internal set; }
        public Texture2D Image { get; internal set; }
        public Pitch Pitch { get; internal set; }
        public Speed Speed { get; internal set; }
        public string Intro { get; internal set; }
        internal static string saveLocation;
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

        public async Task CreateAvatarAsync()
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
            string characterConfigJson = JsonUtility.ToJson(characterConfig);
            File.WriteAllText(Path.Combine(characterFolder, "character.json"), characterConfigJson);

            // Save image (convert to uncompressed format if needed)
            string imagePath = Path.Combine(characterFolder, "image.png");
            var uncompressedImage = LiveTalk.Utils.TextureUtils.ConvertToUncompressedTexture(Image);
            byte[] imageBytes = uncompressedImage.EncodeToPNG();
            await File.WriteAllBytesAsync(imagePath, imageBytes);
            
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
            
            for (int expressionIndex = 0; expressionIndex < expressions.Length; expressionIndex++)
            {
                string expression = expressions[expressionIndex];
                string expressionFolder = Path.Combine(drivingFramesFolder, $"expression-{expressionIndex}");
                Directory.CreateDirectory(expressionFolder);

                Debug.Log($"[Character] Processing expression: {expression} (index: {expressionIndex})");

                try
                {
                    // Load the driving video for this expression
                    VideoClip drivingVideo = LoadDrivingVideoForExpression(expression);
                    if (drivingVideo == null)
                    {
                        Debug.LogWarning($"[Character] Could not load driving video for expression: {expression}");
                        continue;
                    }

                    // Create a temporary VideoPlayer to process the driving frames
                    var tempGO = new GameObject("TempVideoPlayer");
                    try
                    {
                        var videoPlayer = tempGO.AddComponent<VideoPlayer>();
                        videoPlayer.clip = drivingVideo;

                        // Generate animated textures using LivePortrait
                        var outputStream = liveTalkAPI.GenerateAnimatedTexturesAsync(Image, videoPlayer);

                        // Process and save frames as they become available
                        var generatedFrames = new List<Texture2D>();
                        int frameIndex = 0;
                        
                        while (!outputStream.Finished)
                        {
                            if (outputStream.TryGetNext(out Texture2D frame))
                            {
                                // Save frame immediately to disk
                                string frameFileName = Path.Combine(expressionFolder, $"{frameIndex:D5}.png");
                                byte[] pngData = frame.EncodeToPNG();
                                await File.WriteAllBytesAsync(frameFileName, pngData);
                                
                                // Keep reference for cache generation
                                generatedFrames.Add(frame);
                                frameIndex++;
                            }
                            else
                            {
                                await Task.Yield(); // Allow other operations to proceed
                            }
                        }

                        // Collect any remaining frames
                        while (outputStream.TryGetNext(out Texture2D remainingFrame))
                        {
                            string frameFileName = Path.Combine(expressionFolder, $"{frameIndex:D5}.png");
                            byte[] pngData = remainingFrame.EncodeToPNG();
                            await File.WriteAllBytesAsync(frameFileName, pngData);
                            
                            generatedFrames.Add(remainingFrame);
                            frameIndex++;
                        }

                        Debug.Log($"[Character] Generated and saved {generatedFrames.Count} frames for expression: {expression}");

                        // Generate and save cache data
                        await GenerateAndSaveCacheData(expressionFolder, generatedFrames);

                        // Clean up generated textures
                        foreach (var frame in generatedFrames)
                        {
                            UnityEngine.Object.DestroyImmediate(frame);
                        }
                    }
                    finally
                    {
                        UnityEngine.Object.DestroyImmediate(tempGO);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[Character] Error processing expression {expression}: {ex.Message}");
                }
            }

            // Step 3: Generate voice sample using SparkTTS
            await GenerateVoiceSample(voiceFolder);

            Debug.Log($"[Character] Avatar creation completed for {Name}");
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

                // Create a temporary MuseTalkInference instance with the same config as LiveTalkAPI
                var config = new LiveTalk.API.LiveTalkConfig(Application.streamingAssetsPath);
                var museTalkInference = new LiveTalk.Core.MuseTalkInference(config);

                // Convert frames to avatar textures array for processing
                var avatarTextures = frames.ToArray();

                // Use MuseTalkInference to process the avatar images and extract real data
                var avatarData = await ProcessAvatarImagesWithMuseTalk(museTalkInference, avatarTextures);

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

                // Clean up
                museTalkInference?.Dispose();
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
        private async Task<LiveTalk.Core.AvatarData> ProcessAvatarImagesWithMuseTalk(LiveTalk.Core.MuseTalkInference museTalkInference, Texture2D[] avatarTextures)
        {
            Debug.Log($"[Character] Processing {avatarTextures.Length} avatar textures using real MuseTalk pipeline");

            // Use the public MuseTalk ProcessAvatarImages API directly - no reflection needed
            var avatarData = await museTalkInference.ProcessAvatarImages(avatarTextures);
            
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
                
                // Create texture subfolders
                var subfolders = new[]
                {
                    "cropped", "original", "faceLarge", "segmentationMask",
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
                var textureMap = new List<(LiveTalk.Core.Frame frame, string folder, string key)>
                {
                    (face.CroppedFaceTexture, "cropped", "croppedFace"),
                    (face.OriginalTexture, "original", "original"),
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
            try
            {
                // Convert enums to string parameters for SparkTTS
                string genderParam = ConvertGenderToString(Gender);
                string pitchParam = ConvertPitchToString(Pitch);
                string speedParam = ConvertSpeedToString(Speed);

                Debug.Log($"[Character] Generating voice sample with parameters: Gender={genderParam}, Pitch={pitchParam}, Speed={speedParam}");

                // Use SparkTTS CharacterVoiceFactory to create a voice
                var voiceFactory = new CharacterVoiceFactory();
                var characterVoice = await voiceFactory.CreateFromStyleAsync(
                    gender: genderParam,
                    pitch: pitchParam,
                    speed: speedParam,
                    referenceText: Intro ?? "Hello, I am a character in this mystery."
                );

                if (characterVoice != null)
                {
                    // Generate the voice sample
                    var audioClip = await characterVoice.GenerateSpeechAsync(
                        Intro ?? "Hello, I am a character in this mystery."
                    );

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

                voiceFactory.Dispose();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Character] Error generating voice sample: {ex.Message}");
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
        private static bool _initialized = false;

        public static void Initialize(
            string saveLocation,
            LiveTalkController avatarController)
        {
            if (_liveTalkAPI == null && !_initialized)
            {
                _liveTalkAPI = LivePortraitMuseTalkFactory.Create(avatarController);
                Character.saveLocation = saveLocation;
                _initialized = true;
            }
        }

        public static async Task<Character> CreateCharacterAsync(
            string name,
            Gender gender,
            Texture2D image,
            Pitch pitch,
            Speed speed,
            string intro)
        {
            if (!_initialized)
            {
                throw new Exception("CharacterFactory not initialized. Call Initialize() first.");
            }

            var character = new Character(name, gender, image, pitch, speed, intro);
            await character.CreateAvatarAsync();
            return character;
        }
    }
}