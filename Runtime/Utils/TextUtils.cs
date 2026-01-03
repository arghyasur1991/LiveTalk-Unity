using System.Collections.Generic;
using System;
using UnityEngine;
using System.Text;

namespace LiveTalk.Utils
{
    /// <summary>
    /// Utility functions for text processing
    /// </summary>
    public static class TextUtils
    {
        /// <summary>
        /// Generates a consistent ID hash for text to use in audio caching
        /// </summary>
        public static string GenerateTextHash(string text)
        {
            if (string.IsNullOrEmpty(text))
                return "empty";
                
            // Use MD5 to create a consistent hash from the text
            using (var md5 = System.Security.Cryptography.MD5.Create())
            {
                byte[] inputBytes = System.Text.Encoding.UTF8.GetBytes(text);
                byte[] hashBytes = md5.ComputeHash(inputBytes);
                
                // Convert the byte array to hexadecimal string
                StringBuilder sb = new();
                for (int i = 0; i < hashBytes.Length; i++)
                {
                    sb.Append(hashBytes[i].ToString("x2"));
                }
                return sb.ToString();
            }
        }

        /// <summary>
        /// Cleans text for speech by removing unwanted characters and formatting
        /// </summary>
        public static string CleanTextForSpeech(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return text;
                
            // Store original text for comparison
            string originalText = text;
            
            // Remove markdown formatting symbols but only in pairs
            // This ensures we don't remove single asterisks or underscores
            text = System.Text.RegularExpressions.Regex.Replace(text, @"\*\*(.*?)\*\*|\*(.*?)\*|__(.*?)__|_(.*?)_|~~(.*?)~~|`(.*?)`", "$1$2$3$4$5$6");
            
            // Remove all text between < and >
            text = System.Text.RegularExpressions.Regex.Replace(text, @"<[^>]*>", "");
            
            // Remove only specific problematic symbols, preserving most punctuation
            text = System.Text.RegularExpressions.Regex.Replace(text, @"[\\|()@*#$%^&{}]", "");
            
            // Replace multiple spaces with a single space
            text = System.Text.RegularExpressions.Regex.Replace(text, @"\s+", " ");

            // Trim whitespace
            text = text.Trim();
            
            // Log before and after for debugging if needed
            // Debug.Log($"Original text: {originalText}");
            // Debug.Log($"Cleaned text: {text}");
            
            return text;
        }

        /// <summary>
        /// Breaks text into individual sentences
        /// </summary>
        public static string[] BreakTextIntoLines(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new string[0];

            // Common abbreviations that contain periods but shouldn't be split
            HashSet<string> commonAbbreviations = new()
            {
                "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Rev.", "Sr.", "Jr.", "Ph.D.", "M.D.", "B.A.", "B.S.",
                "i.e.", "e.g.", "etc.", "vs.", "a.m.", "p.m.", "U.S.", "U.K.", "Fig."
            };

            // Step 1: Split by sentence endings but preserve the delimiter
            List<string> sentences = new();
            int startPos = 0;
            bool inQuotes = false;
            int inParentheses = 0;

            for (int i = 0; i < text.Length; i++)
            {
                // Track if we're inside quotes or parentheses
                if (text[i] == '"' || text[i] == '"' || text[i] == '"')
                    inQuotes = !inQuotes;
                else if (text[i] == '(')
                    inParentheses++;
                else if (text[i] == ')')
                    inParentheses = Math.Max(0, inParentheses - 1);

                // Handle newlines
                if (text[i] == '\n' || text[i] == '\r')
                {
                    // Handle Windows-style line endings (\r\n)
                    int endPos = i + 1;
                    if (text[i] == '\r' && i + 1 < text.Length && text[i + 1] == '\n')
                    {
                        endPos = i + 2;
                        i++; // Skip the \n
                    }
                    
                    string sentence = text[startPos..i].Trim();
                    sentence = CleanTextForSpeech(sentence);
                    
                    // Only add non-empty sentences
                    if (!string.IsNullOrWhiteSpace(sentence))
                    {
                        sentences.Add(sentence);
                    }
                    
                    startPos = endPos;
                    continue;
                }

                // Handle HTML tag endings
                if (text[i] == '>' && i >= 2)
                {
                    // Look backwards to find if this is a closing tag
                    bool isClosingTag = false;
                    int tagStart = -1;
                    
                    // Find the opening < of this tag
                    for (int j = i - 1; j >= 0; j--)
                    {
                        if (text[j] == '<')
                        {
                            tagStart = j;
                            // Check if it's a closing tag (starts with </)
                            if (j + 1 < text.Length && text[j + 1] == '/')
                            {
                                isClosingTag = true;
                            }
                            break;
                        }
                        // If we hit another > before finding <, this isn't a valid tag
                        if (text[j] == '>')
                            break;
                    }
                    
                    if (isClosingTag && tagStart >= startPos)
                    {
                        // End of HTML closing tag found
                        int endPos = i + 1;
                        string sentence = text[startPos..endPos].Trim();
                        sentence = CleanTextForSpeech(sentence);
                        
                        // Only add non-empty sentences
                        if (!string.IsNullOrWhiteSpace(sentence))
                        {
                            sentences.Add(sentence);
                        }
                        
                        startPos = endPos;
                        continue;
                    }
                }

                // Handle periods
                if (text[i] == '.' || text[i] == '?' || text[i] == '!')
                {
                    // Check if this is part of an ellipsis
                    if (text[i] == '.' && i + 2 < text.Length && text[i + 1] == '.' && text[i + 2] == '.')
                    {
                        // Skip to the end of the ellipsis
                        i += 2;
                        continue;
                    }

                    // Check if this is part of a common abbreviation
                    bool isAbbreviation = false;
                    foreach (string abbr in commonAbbreviations)
                    {
                        if (i + 1 >= abbr.Length && 
                            i + 1 < text.Length && 
                            text.Substring(i - abbr.Length + 1, abbr.Length).Equals(abbr, StringComparison.OrdinalIgnoreCase) &&
                            (i + 1 == text.Length || char.IsWhiteSpace(text[i + 1])))
                        {
                            isAbbreviation = true;
                            break;
                        }
                    }

                    if (isAbbreviation)
                        continue;

                    // End of sentence found
                    int endPos = i + 1;
                    string sentence = text[startPos..endPos].Trim();

                    sentence = CleanTextForSpeech(sentence);
                    
                    // Only add non-empty sentences
                    if (!string.IsNullOrWhiteSpace(sentence))
                    {
                        sentences.Add(sentence);
                    }
                    
                    startPos = endPos;
                }
            }

            // Add the last sentence if there's any text left
            if (startPos < text.Length)
            {
                string lastSentence = text[startPos..].Trim();
                lastSentence = CleanTextForSpeech(lastSentence);
                if (!string.IsNullOrWhiteSpace(lastSentence))
                {
                    sentences.Add(lastSentence);
                }
            }

            // If no sentences were found (e.g., short text without punctuation), add the entire text as one sentence
            if (sentences.Count == 0 && !string.IsNullOrWhiteSpace(text))
            {
                string cleanedText = CleanTextForSpeech(text);
                if (!string.IsNullOrWhiteSpace(cleanedText))
                {
                    sentences.Add(cleanedText);
                }
            }

            return sentences.ToArray();
        }
    }
}

