using System;
using System.IO;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

namespace LiveTalk.Editor
{
    /// <summary>
    /// CUDA in the Windows editor. <c>com.github.asus4.onnxruntime</c> 0.4.9
    /// ships the CPU <c>onnxruntime.dll</c> for Windows x64. The CUDA execution
    /// provider registers against it and opens sessions, but the first cuDNN
    /// convolution in <c>Session.Run</c> fails fast. The core from
    /// <c>Microsoft.ML.OnnxRuntime.Gpu.Windows</c> 1.29.0 (same ORT revision,
    /// with the CUDA host hooks) runs.
    ///
    /// <para>
    /// Point LiveTalk at that file — <c>runtimes/win-x64/native/onnxruntime.dll</c>
    /// from the NuGet package — with <b>LiveTalk → Windows GPU Core…</b>, the
    /// <see cref="SourcePrefKey"/> editor pref, or the
    /// <see cref="SourceEnvVar"/> environment variable (a file or its folder).
    /// On editor load the overlay swaps it in for the package's CPU core,
    /// only when both files match the hashes below, and asks for a restart:
    /// the old core is already mapped into the process.
    /// </para>
    /// <para>
    /// Without a configured source nothing happens, and the console stays
    /// quiet unless the CUDA provider package is installed.
    /// </para>
    /// </summary>
    public static class WindowsGpuCoreOverlay
    {
        /// <summary>EditorPrefs key holding the GPU core path (file or folder).</summary>
        public const string SourcePrefKey = "LiveTalk.WindowsGpuCore";

        /// <summary>Environment variable consulted when the pref is empty.</summary>
        public const string SourceEnvVar = "LIVETALK_ORT_GPU_CORE";

        const string CoreFileName = "onnxruntime.dll";

        // Microsoft.ML.OnnxRuntime.Gpu.Windows 1.29.0, runtimes/win-x64/native.
        const string GpuCoreSha256 = "5458C46E26EFE64D7B2F960BA6AFF97209B454A007AF0F93D682AC2570F7541D";
        const long GpuCoreBytes = 16588600;

        // com.github.asus4.onnxruntime 0.4.9 (ORT 1.29.0) win-x64 CPU core:
        // the only file this overlay will replace.
        const string CpuCoreSha256 = "69D8E6D3879A3B4001CDC74C8ED9CCC7E7F799A5B847059738323404519EC471";

        /// <summary>
        /// The configured GPU core, or null. The pref wins over the
        /// environment variable; either may name the DLL or its folder.
        /// </summary>
        public static string ConfiguredSource
        {
            get
            {
                string value = EditorPrefs.GetString(SourcePrefKey, "");
                if (string.IsNullOrEmpty(value))
                    value = Environment.GetEnvironmentVariable(SourceEnvVar);
                if (string.IsNullOrEmpty(value))
                    return null;
                if (Directory.Exists(value))
                    value = Path.Combine(value, CoreFileName);
                return File.Exists(value) ? value : null;
            }
        }

        /// <summary>
        /// Copies the configured GPU core over the package's CPU core when
        /// that core is still the known CPU build. Returns true when the file
        /// on disk changed; ONNX Runtime must not be created until the next
        /// editor launch. <paramref name="message"/> is set when there is
        /// something to tell the user.
        /// </summary>
        internal static bool TryApply(out string message)
        {
            message = null;
#if !UNITY_EDITOR_WIN
            return false;
#else
            string installed = FindInstalledCore();
            if (string.IsNullOrEmpty(installed))
                return false;

            string installedHash = Sha256(installed);
            if (string.Equals(installedHash, GpuCoreSha256, StringComparison.OrdinalIgnoreCase))
                return false;
            if (!string.Equals(installedHash, CpuCoreSha256, StringComparison.OrdinalIgnoreCase))
            {
                message = "Left the Windows onnxruntime.dll alone: it is not the known 1.29.0 CPU build (" + installedHash + ").";
                return false;
            }

            string source = ConfiguredSource;
            if (string.IsNullOrEmpty(source))
            {
                if (CudaProviderPackageInstalled())
                {
                    message = "CUDA needs the Microsoft.ML.OnnxRuntime.Gpu.Windows 1.29.0 onnxruntime.dll. " +
                              "Set it with LiveTalk → Windows GPU Core… (or " + SourceEnvVar + "). Running on the CPU core until then.";
                }
                return false;
            }

            var info = new FileInfo(source);
            string sourceHash = Sha256(source);
            if (info.Length != GpuCoreBytes
                || !string.Equals(sourceHash, GpuCoreSha256, StringComparison.OrdinalIgnoreCase))
            {
                message = "Ignored " + source + ": not the Gpu.Windows 1.29.0 onnxruntime.dll (" + sourceHash + ").";
                return false;
            }

            string backup = installed + ".cpu-1.29.0";
            if (File.Exists(backup))
                File.Delete(backup);
            File.Move(installed, backup);
            File.Copy(source, installed, overwrite: false);
            if (!string.Equals(Sha256(installed), GpuCoreSha256, StringComparison.OrdinalIgnoreCase))
            {
                File.Delete(installed);
                File.Move(backup, installed);
                message = "GPU core copy failed its hash check; restored the CPU core.";
                return false;
            }
            File.Delete(backup);
            message = "Installed the Gpu.Windows 1.29.0 onnxruntime.dll. Restart the editor before creating an ONNX session.";
            return true;
#endif
        }

#if UNITY_EDITOR_WIN
        [MenuItem("LiveTalk/Windows GPU Core…")]
        static void PickSource()
        {
            string current = ConfiguredSource;
            string picked = EditorUtility.OpenFilePanel(
                "Gpu.Windows 1.29.0 onnxruntime.dll",
                string.IsNullOrEmpty(current) ? "" : Path.GetDirectoryName(current),
                "dll");
            if (string.IsNullOrEmpty(picked))
                return;
            EditorPrefs.SetString(SourcePrefKey, picked);
            if (TryApply(out string message))
                EditorUtility.DisplayDialog("LiveTalk", message, "OK");
            else if (!string.IsNullOrEmpty(message))
                Debug.LogWarning("[LiveTalk] " + message);
        }
#endif

        static string ProjectPackageCache()
        {
            string project = Path.GetDirectoryName(Application.dataPath);
            if (string.IsNullOrEmpty(project))
                return null;
            string cache = Path.Combine(project, "Library", "PackageCache");
            return Directory.Exists(cache) ? cache : null;
        }

        static string FindInstalledCore()
        {
            string cache = ProjectPackageCache();
            if (cache == null)
                return null;
            foreach (string pkg in Directory.GetDirectories(cache, "com.github.asus4.onnxruntime@*"))
            {
                string dll = Path.Combine(pkg, "Plugins", "Windows", "x64", CoreFileName);
                if (File.Exists(dll))
                    return dll;
            }
            return null;
        }

        static bool CudaProviderPackageInstalled()
        {
            string cache = ProjectPackageCache();
            return cache != null
                && Directory.GetDirectories(cache, "com.github.asus4.onnxruntime.win-x64-gpu@*").Length > 0;
        }

        static string Sha256(string path)
        {
            using var stream = File.OpenRead(path);
            using var sha = SHA256.Create();
            byte[] hash = sha.ComputeHash(stream);
            return BitConverter.ToString(hash).Replace("-", "");
        }
    }
}
