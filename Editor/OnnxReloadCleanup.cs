using System;
using LiveTalk.API;
using Microsoft.ML.OnnxRuntime;
using UnityEditor;
using UnityEngine;

namespace LiveTalk.Editor
{
    /// <summary>
    /// Disposes every LiveTalk inference session before a domain reload.
    ///
    /// ONNX Runtime allows one environment per process, and the TTS package
    /// releases it in its own <c>beforeAssemblyReload</c> hook. A LiveTalk
    /// <c>InferenceSession</c> still alive at that point is torn down by the
    /// finalizer thread during domain unload, against an environment that no
    /// longer exists — a native access violation in <c>OrtReleaseSession</c>
    /// that takes the editor with it. Disposing here, while the environment
    /// is intact, is the same order the TTS package uses for its own sessions.
    ///
    /// A bake in flight is lost; the alternative is a crash. Hosts that drive
    /// LiveTalk from edit-mode code should cancel their pump before editing
    /// scripts, because a script change is a reload.
    /// </summary>
    [InitializeOnLoad]
    static class OnnxReloadCleanup
    {
        static OnnxReloadCleanup()
        {
            AssemblyReloadEvents.beforeAssemblyReload += ReleaseBeforeReload;
        }

        static void ReleaseBeforeReload()
        {
            try
            {
                if (!OrtEnv.IsCreated)
                {
                    // The environment is already gone; disposing now would hit the
                    // same dead pointer the finalizer does. Nothing safe remains.
                    Debug.LogWarning("[LiveTalk] ONNX environment released before LiveTalk sessions; check reload-hook order.");
                    return;
                }
                LiveTalkAPI.Instance?.Dispose();
                Debug.Log("[LiveTalk] Released ONNX sessions before reload.");
            }
            catch (Exception e)
            {
                Debug.LogWarning("[LiveTalk] Pre-reload cleanup: " + e.Message);
            }
        }
    }
}
