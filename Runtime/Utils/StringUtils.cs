namespace LiveTalk.Utils
{
    /// <summary>
    /// String utilities for LiveTalk face animation and geometric processing.
    /// Provides optimized implementations of string operations, hash mixing, and string formatting
    /// </summary>
    public static class StringUtils
    {
        /// <summary>
        /// Mix multiple hash strings into a single deterministic hash using FNV-1a-like algorithm
        /// </summary>
        /// <param name="hashes">Array of hash strings in hex format</param>
        /// <returns>Combined hash as 8-character hex string</returns>
        public static string MixHash(params string[] hashes)
        {
            if (hashes == null || hashes.Length == 0)
                return "00000000";
            
            uint combinedHash = 0x811C9DC5; // FNV-1a offset basis (32-bit)
            
            // Mix each hash using FNV-1a-like algorithm for deterministic results
            foreach (string hash in hashes)
            {
                if (!string.IsNullOrEmpty(hash))
                {
                    // Convert hex string to bytes and mix each byte
                    for (int i = 0; i < hash.Length; i += 2)
                    {
                        if (i + 1 < hash.Length)
                        {
                            string byteStr = hash.Substring(i, 2);
                            if (byte.TryParse(byteStr, System.Globalization.NumberStyles.HexNumber, null, out byte b))
                            {
                                combinedHash ^= b;
                                combinedHash *= 0x01000193; // FNV-1a prime (32-bit)
                            }
                        }
                    }
                }
            }
            
            // Return as 8-character uppercase hex string
            return combinedHash.ToString("X8");
        }
    }

} 