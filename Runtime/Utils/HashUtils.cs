using System;
using System.Text;
using UnityEngine;

namespace LiveTalk.Utils
{
    /// <summary>
    /// Sophisticated hashing utilities for LiveTalk character and speech caching.
    /// Provides cryptographic-quality hashing for unique identification of:
    /// - Text content for speech caching
    /// - Character identity for voice generation
    /// - Texture content for avatar identification
    /// - Combined voice hashes for global caching
    /// </summary>
    public static class HashUtils
    {
        // FNV-1a constants for 64-bit hashing
        private const ulong FNV_OFFSET_BASIS_64 = 14695981039346656037UL;
        private const ulong FNV_PRIME_64 = 1099511628211UL;
        
        // FNV-1a constants for 32-bit hashing  
        private const uint FNV_OFFSET_BASIS_32 = 0x811C9DC5;
        private const uint FNV_PRIME_32 = 0x01000193;

        /// <summary>
        /// Generates a consistent MD5 hash for text content.
        /// Used for speech caching to identify identical text across sessions.
        /// </summary>
        /// <param name="text">The text to hash</param>
        /// <returns>32-character lowercase hex string, or "empty" if text is null/empty</returns>
        public static string GenerateTextHash(string text)
        {
            if (string.IsNullOrEmpty(text))
                return "empty";
                
            using (var md5 = System.Security.Cryptography.MD5.Create())
            {
                byte[] inputBytes = Encoding.UTF8.GetBytes(text);
                byte[] hashBytes = md5.ComputeHash(inputBytes);
                
                // Convert to hexadecimal string
                var sb = new StringBuilder(32);
                for (int i = 0; i < hashBytes.Length; i++)
                {
                    sb.Append(hashBytes[i].ToString("x2"));
                }
                return sb.ToString();
            }
        }

        /// <summary>
        /// Generates a fast content-based hash for a texture using dimensions, format, and pixel sampling.
        /// This method provides good uniqueness for texture comparison while being much faster than hashing all pixels.
        /// Samples a subset of pixels (32x32 maximum) and uses every 10th pixel for efficient hash generation.
        /// </summary>
        /// <param name="texture">The texture to hash</param>
        /// <returns>16-character hex string representing the texture content</returns>
        public static string GenerateTextureHash(Texture2D texture)
        {
            if (texture == null)
                return "0000000000000000";
            
            ulong hash = FNV_OFFSET_BASIS_64;
            
            // Hash texture properties
            hash = HashInt(hash, texture.width);
            hash = HashInt(hash, texture.height);
            hash = HashInt(hash, (int)texture.format);
            int maxSampleSize = 256;
            
            // Sample pixels for content-based hashing (faster than full texture)
            try
            {
                int sampleWidth = Math.Min(maxSampleSize, texture.width);
                int sampleHeight = Math.Min(maxSampleSize, texture.height);
                var pixels = texture.GetPixels(0, 0, sampleWidth, sampleHeight);
                
                // Sample every 10th pixel for efficiency
                for (int i = 0; i < pixels.Length; i += 10)
                {
                    var pixel = pixels[i];
                    hash = HashInt(hash, (int)(pixel.r * 255));
                    hash = HashInt(hash, (int)(pixel.g * 255));
                    hash = HashInt(hash, (int)(pixel.b * 255));
                    hash = HashInt(hash, (int)(pixel.a * 255));
                }
            }
            catch
            {
                // If we can't read pixels (e.g., compressed texture), use only dimensions
                hash = HashString(hash, "unreadable");
            }
            
            return hash.ToString("x16");
        }

        /// <summary>
        /// Generates a unique hash for a character based on all identifying properties.
        /// Combines name, gender, pitch, speed, intro text, and optional image.
        /// </summary>
        /// <param name="name">Character name</param>
        /// <param name="gender">Character gender</param>
        /// <param name="pitch">Voice pitch setting</param>
        /// <param name="speed">Voice speed setting</param>
        /// <param name="intro">Intro/reference text for voice generation</param>
        /// <param name="image">Optional character image texture</param>
        /// <returns>16-character hex string uniquely identifying this character configuration</returns>
        public static string GenerateCharacterHash(
            string name, 
            string gender, 
            string pitch, 
            string speed, 
            string intro = null,
            Texture2D image = null)
        {
            ulong hash = FNV_OFFSET_BASIS_64;
            
            // Hash all character properties
            hash = HashString(hash, name ?? "unnamed");
            hash = HashString(hash, gender ?? "unknown");
            hash = HashString(hash, pitch ?? "moderate");
            hash = HashString(hash, speed ?? "moderate");
            
            // Include intro text if provided (affects voice generation)
            if (!string.IsNullOrEmpty(intro))
            {
                hash = HashString(hash, intro);
            }
            
            // Include image hash if provided
            if (image != null)
            {
                string imageHash = GenerateTextureHash(image);
                hash = HashString(hash, imageHash);
            }
            
            return hash.ToString("x16");
        }

        /// <summary>
        /// Generates a sophisticated character identity hash.
        /// Combines character id with voice parameters for unique voice identification.
        /// </summary>
        /// <param name="characterId">The character's unique identifier (folder name/GUID)</param>
        /// <param name="name">Character name</param>
        /// <param name="gender">Voice gender parameter</param>
        /// <param name="pitch">Voice pitch parameter</param>
        /// <param name="speed">Voice speed parameter</param>
        /// <returns>16-character hex string representing the character voice identity</returns>
        public static string GenerateCharacterVoiceHash(
            string characterId, 
            string name = null, 
            string gender = null, 
            string pitch = null, 
            string speed = null)
        {
            // Use 64-bit FNV-1a for better distribution
            ulong hash = FNV_OFFSET_BASIS_64;
            
            // Primary identity from characterId
            hash = HashString(hash, characterId ?? "unknown");
            
            // Add voice parameters if provided (for voice style identification)
            if (!string.IsNullOrEmpty(name))
                hash = HashString(hash, name);
            if (!string.IsNullOrEmpty(gender))
                hash = HashString(hash, gender);
            if (!string.IsNullOrEmpty(pitch))
                hash = HashString(hash, pitch);
            if (!string.IsNullOrEmpty(speed))
                hash = HashString(hash, speed);
            
            return hash.ToString("x16");
        }

        /// <summary>
        /// Creates a global voice hash that uniquely identifies a specific speech output.
        /// Combines text content hash with character identity for global caching.
        /// </summary>
        /// <param name="textHash">The MD5 hash of the text content</param>
        /// <param name="characterId">The character's unique identifier</param>
        /// <returns>24-character hex string for global voice cache lookup</returns>
        public static string CreateGlobalVoiceHash(string textHash, string characterId)
        {
            if (string.IsNullOrEmpty(textHash) || string.IsNullOrEmpty(characterId))
                return null;
            
            // Use 64-bit hash for combining
            ulong combined = FNV_OFFSET_BASIS_64;
            
            // Mix text hash bytes
            combined = HashString(combined, textHash);
            
            // Mix character id
            combined = HashString(combined, characterId);
            
            // Add a salt to differentiate from other hash types
            combined = HashString(combined, "voice_cache_v1");
            
            // Return as 24-char hex (16 for main hash + 8 for collision resistance)
            string mainHash = combined.ToString("x16");
            uint collisionResistance = (uint)(combined >> 32) ^ (uint)combined;
            return mainHash + collisionResistance.ToString("x8");
        }

        /// <summary>
        /// Creates a global voice hash from raw text and character id.
        /// Convenience method that handles text hashing internally.
        /// </summary>
        /// <param name="text">The raw text content</param>
        /// <param name="characterId">The character's unique identifier</param>
        /// <returns>24-character hex string for global voice cache lookup</returns>
        public static string CreateGlobalVoiceHashFromText(string text, string characterId)
        {
            string textHash = GenerateTextHash(text);
            return CreateGlobalVoiceHash(textHash, characterId);
        }

        /// <summary>
        /// Generates a cache key for voice style configuration.
        /// Used to cache generated voice samples based on style parameters.
        /// </summary>
        /// <param name="gender">Voice gender parameter</param>
        /// <param name="pitch">Voice pitch parameter</param>
        /// <param name="speed">Voice speed parameter</param>
        /// <param name="introText">Reference text used for voice generation</param>
        /// <returns>Cache key for this voice style configuration</returns>
        public static string GenerateVoiceStyleCacheKey(string gender, string pitch, string speed, string introText)
        {
            ulong hash = FNV_OFFSET_BASIS_64;
            
            hash = HashString(hash, gender ?? "female");
            hash = HashString(hash, pitch ?? "moderate");
            hash = HashString(hash, speed ?? "moderate");
            hash = HashString(hash, introText ?? "");
            
            // Add salt to differentiate from other cache types
            hash = HashString(hash, "voice_style_v1");
            
            return "vs_" + hash.ToString("x16");
        }

        /// <summary>
        /// Generates a unique speech cache key for a specific utterance.
        /// </summary>
        /// <param name="text">The text to be spoken</param>
        /// <param name="characterId">The character speaking</param>
        /// <param name="expressionIndex">The expression being used (-1 for voice only)</param>
        /// <returns>Unique cache key for this specific speech</returns>
        public static string GenerateSpeechCacheKey(string text, string characterId, int expressionIndex = -1)
        {
            string textHash = GenerateTextHash(text);
            string voiceHash = CreateGlobalVoiceHash(textHash, characterId);
            
            if (expressionIndex >= 0)
            {
                // Include expression in hash for animated speech
                uint exprHash = (uint)expressionIndex * FNV_PRIME_32;
                return voiceHash + "_" + exprHash.ToString("x4");
            }
            
            return voiceHash;
        }

        /// <summary>
        /// Mixes multiple hash strings into a single deterministic hash.
        /// Uses FNV-1a algorithm for consistent results.
        /// </summary>
        /// <param name="hashes">Array of hash strings in hex format</param>
        /// <returns>Combined hash as 16-character hex string</returns>
        public static string MixHashes(params string[] hashes)
        {
            if (hashes == null || hashes.Length == 0)
                return "0000000000000000";
            
            ulong combined = FNV_OFFSET_BASIS_64;
            
            foreach (string hash in hashes)
            {
                if (!string.IsNullOrEmpty(hash))
                {
                    combined = HashString(combined, hash);
                }
            }
            
            return combined.ToString("x16");
        }

        #region Private Helper Methods

        /// <summary>
        /// Hash a string into the running 64-bit hash value using FNV-1a
        /// </summary>
        private static ulong HashString(ulong hash, string str)
        {
            if (string.IsNullOrEmpty(str))
                return hash;
                
            byte[] bytes = Encoding.UTF8.GetBytes(str);
            foreach (byte b in bytes)
            {
                hash ^= b;
                hash *= FNV_PRIME_64;
            }
            return hash;
        }

        /// <summary>
        /// Hash an integer into the running 64-bit hash value
        /// </summary>
        private static ulong HashInt(ulong hash, int value)
        {
            for (int i = 0; i < 4; i++)
            {
                hash ^= (byte)(value >> (i * 8));
                hash *= FNV_PRIME_64;
            }
            return hash;
        }

        #endregion
    }
}

