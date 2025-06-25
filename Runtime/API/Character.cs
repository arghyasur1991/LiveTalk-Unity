using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

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
            // TODO: Create avatar
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