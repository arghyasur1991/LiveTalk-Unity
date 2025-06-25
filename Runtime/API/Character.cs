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
            
            // Create main character folder
            Directory.CreateDirectory(characterFolder);
            
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

                        // Wait for processing to complete and collect frames
                        var generatedFrames = new List<Texture2D>();
                        
                        while (!outputStream.Finished)
                        {
                            if (outputStream.TryGetNext(out Texture2D frame))
                            {
                                generatedFrames.Add(frame);
                            }
                            await Task.Yield(); // Allow other operations to proceed
                        }

                        // Collect any remaining frames
                        while (outputStream.TryGetNext(out Texture2D remainingFrame))
                        {
                            generatedFrames.Add(remainingFrame);
                        }

                        Debug.Log($"[Character] Generated {generatedFrames.Count} frames for expression: {expression}");

                        // Save generated frames as PNG files
                        for (int i = 0; i < generatedFrames.Count; i++)
                        {
                            string frameFileName = Path.Combine(expressionFolder, $"{i:D5}.png");
                            byte[] pngData = generatedFrames[i].EncodeToPNG();
                            await File.WriteAllBytesAsync(frameFileName, pngData);
                        }

                        // Generate and save latents and face data for caching
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
            string imageHash = GenerateImageHash();
            
            return $"{nameHash}_{genderHash}_{imageHash}";
        }

        /// <summary>
        /// Generate a hash for the character image
        /// </summary>
        private string GenerateImageHash()
        {
            if (Image == null) 
                return "00000000";

            unchecked
            {
                int hash = Image.width.GetHashCode();
                hash = hash * 31 + Image.height.GetHashCode();
                hash = hash * 31 + Image.format.GetHashCode();
                
                // Sample a few pixels for content-based hashing
                var pixels = Image.GetPixels(0, 0, Math.Min(32, Image.width), Math.Min(32, Image.height));
                for (int i = 0; i < Math.Min(100, pixels.Length); i += 10)
                {
                    hash = hash * 31 + pixels[i].GetHashCode();
                }
                
                return hash.ToString("X8");
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
                    Debug.Log($"[Character] Loaded driving video: {path}");
                    return videoClip;
                }
            }

            Debug.LogWarning($"[Character] Could not find driving video for expression: {expression}");
            return null;
        }

        /// <summary>
        /// Generate and save cache data (latents and face data) for the processed frames
        /// </summary>
        private async Task GenerateAndSaveCacheData(string expressionFolder, List<Texture2D> frames)
        {
            try
            {
                // This would typically involve running the frames through MuseTalk's avatar processing
                // to generate latents and face data that can be cached for faster future processing
                
                // For now, we'll create placeholder cache files
                // In a full implementation, this would extract latents using the VAE encoder
                // and face data using the face analysis components
                
                var latentsFile = Path.Combine(expressionFolder, "latents.bin");
                var facesFile = Path.Combine(expressionFolder, "faces.json");
                
                // Create placeholder latents (in a real implementation, this would be the actual latents)
                var placeholderLatents = new float[frames.Count * 8 * 32 * 32]; // Example dimensions
                var latentsBytes = new byte[placeholderLatents.Length * sizeof(float)];
                Buffer.BlockCopy(placeholderLatents, 0, latentsBytes, 0, latentsBytes.Length);
                await File.WriteAllBytesAsync(latentsFile, latentsBytes);
                
                // Create placeholder face data (in a real implementation, this would contain face landmarks and regions)
                var faceData = new
                {
                    frames = frames.Count,
                    timestamp = DateTime.UtcNow,
                    version = "1.0"
                };
                
                string faceDataJson = Newtonsoft.Json.JsonConvert.SerializeObject(faceData, Newtonsoft.Json.Formatting.Indented);
                await File.WriteAllTextAsync(facesFile, faceDataJson);
                
                Debug.Log($"[Character] Generated cache data for {frames.Count} frames");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Character] Error generating cache data: {ex.Message}");
            }
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