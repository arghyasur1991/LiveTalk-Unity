using LiveTalk.API;
using UnityEditor;
using UnityEngine;

namespace LiveTalk.Editor
{
    /// <summary>
    /// Editor load runs before any window or bake can create OrtEnv, so the
    /// CUDA provider folder is on PATH for the rest of the session.
    /// </summary>
    [InitializeOnLoad]
    static class NativeProviderSearchPath
    {
        static NativeProviderSearchPath()
        {
            // A just-applied overlay is on disk only. The CPU core is already
            // mapped, so do not create OrtEnv until the next launch.
            if (WindowsGpuCoreOverlay.TryApply(out string overlay))
            {
                Debug.LogWarning("[LiveTalk] " + overlay);
                return;
            }
            if (!string.IsNullOrEmpty(overlay))
                Debug.LogWarning("[LiveTalk] " + overlay);
            LiveTalkAPI.PrepareNativeExecutionProviders();
        }

        [MenuItem("LiveTalk/Log ONNX Execution Providers")]
        static void LogProviders()
        {
            LiveTalkAPI.PrepareNativeExecutionProviders();
            string[] providers = LiveTalkAPI.GetAvailableExecutionProviders();
            Debug.Log("[LiveTalk] ONNX providers: " + string.Join(", ", providers));
        }
    }
}
