using UnityEngine;

namespace LiveTalk.Utils
{
    using API;
    /// <summary>
    /// Logger for LiveTalk pipeline
    /// </summary>
    internal static class Logger
    {
        public static LogLevel LogLevel { get; set; } = LogLevel.INFO;

        public static void LogVerbose(string message)
        {
            if (LogLevel <= LogLevel.VERBOSE)
            {
                Debug.Log($"[LiveTalk-VERBOSE] {message}");
            }
        }
        
        public static void Log(string message)
        {
            if (LogLevel <= LogLevel.INFO)
            {
                Debug.Log($"[LiveTalk-INFO] {message}");
            }
        }
        
        public static void LogWarning(string message)
        {
            if (LogLevel <= LogLevel.WARNING)
            {
                Debug.LogWarning($"[LiveTalk-WARNING] {message}");
            }
        }
        
        public static void LogError(string message)
        {
            if (LogLevel <= LogLevel.ERROR)
            {
                Debug.LogError($"[LiveTalk-ERROR] {message}");
            }
        }
    }
}
