using System;
using System.IO;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

namespace LiveTalk.Editor
{
    /// <summary>
    /// asus4 onnxruntime-unity 0.4.9 ships the CPU <c>onnxruntime.dll</c> on
    /// Windows x64. Registering the CUDA EP against that core lets a session
    /// open, then the first cuDNN conv <c>Session.Run</c> fast-fails.
    /// The Microsoft Gpu.Windows 1.29.0 core (same ORT git, CUDA host hooks
    /// included) lives on the Genesis Drive mount. This copies it over the
    /// PackageCache plugin before any session is created.
    /// </summary>
    static class WindowsGpuCoreOverlay
    {
        const string RelativeDll = "onnxruntime/win-x64/1.29.0/onnxruntime.dll";

        // Microsoft.ML.OnnxRuntime.Gpu.Windows 1.29.0 win-x64 core.
        const string GpuCoreSha256 = "5458C46E26EFE64D7B2F960BA6AFF97209B454A007AF0F93D682AC2570F7541D";
        const long GpuCoreBytes = 16588600;

        // asus4 / Microsoft.ML.OnnxRuntime 1.29.0 win-x64 CPU core. The only
        // file this overlay is allowed to replace.
        const string CpuCoreSha256 = "69D8E6D3879A3B4001CDC74C8ED9CCC7E7F799A5B847059738323404519EC471";

        /// <summary>
        /// Copies the Drive GPU core over the Windows x64 CPU plugin when
        /// that plugin is still the known 1.29.0 CPU build. Returns true
        /// when the file on disk changed; the process already mapped the
        /// old DLL, so ORT must not be created until the next editor launch.
        /// </summary>
        internal static bool TryApply(out string message)
        {
            message = null;
#if !UNITY_EDITOR_WIN
            return false;
#else
            string installed = FindInstalledCore();
            if (string.IsNullOrEmpty(installed))
            {
                message = "Windows onnxruntime.dll not in PackageCache yet";
                return false;
            }

            string installedHash = Sha256(installed);
            if (string.Equals(installedHash, GpuCoreSha256, StringComparison.OrdinalIgnoreCase))
            {
                string leftover = installed + ".cpu-1.29.0";
                if (File.Exists(leftover))
                {
                    try { File.Delete(leftover); }
                    catch (IOException) { }
                }
                return false;
            }
            if (!string.Equals(installedHash, CpuCoreSha256, StringComparison.OrdinalIgnoreCase))
            {
                message = "left Windows onnxruntime.dll alone (not the known 1.29.0 CPU build): " + installedHash;
                return false;
            }

            string source = FindDriveCore();
            if (string.IsNullOrEmpty(source))
            {
                message = "Genesis Drive GPU core missing. Expected onnxruntime/win-x64/1.29.0/onnxruntime.dll under My Private/Projects/GenesisInteractive";
                return false;
            }

            var info = new FileInfo(source);
            string sourceHash = Sha256(source);
            if (info.Length != GpuCoreBytes
                || !string.Equals(sourceHash, GpuCoreSha256, StringComparison.OrdinalIgnoreCase))
            {
                message = "refusing Drive GPU core; hash/size is not Gpu.Windows 1.29.0: " + sourceHash;
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
                message = "GPU core copy failed hash check; restored the CPU dll";
                return false;
            }
            File.Delete(backup);
            message = "Replaced the Windows CPU onnxruntime.dll with the Gpu.Windows 1.29.0 core from Drive. Restart the editor before creating an ONNX session.";
            return true;
#endif
        }

        static string FindInstalledCore()
        {
            string project = Path.GetDirectoryName(Application.dataPath);
            if (string.IsNullOrEmpty(project))
                return null;
            string cache = Path.Combine(project, "Library", "PackageCache");
            if (!Directory.Exists(cache))
                return null;
            foreach (string pkg in Directory.GetDirectories(cache, "com.github.asus4.onnxruntime@*"))
            {
                if (pkg.IndexOf("win-x64-gpu", StringComparison.OrdinalIgnoreCase) >= 0)
                    continue;
                string dll = Path.Combine(pkg, "Plugins", "Windows", "x64", "onnxruntime.dll");
                if (File.Exists(dll))
                    return dll;
            }
            return null;
        }

        static string FindDriveCore()
        {
            string root = FindGenesisRoot();
            if (string.IsNullOrEmpty(root))
                return null;
            string dll = Path.Combine(root, RelativeDll.Replace('/', Path.DirectorySeparatorChar));
            return File.Exists(dll) ? dll : null;
        }

        static string FindGenesisRoot()
        {
            string fromEnv = Environment.GetEnvironmentVariable("GENESIS_PROJECTS");
            if (!string.IsNullOrEmpty(fromEnv) && Directory.Exists(fromEnv))
                return fromEnv;

            string windows = Path.Combine(
                "G:", "My Drive", "My Private", "Projects", "GenesisInteractive");
            if (Directory.Exists(windows))
                return windows;

            string cloud = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "Library", "CloudStorage");
            if (!Directory.Exists(cloud))
                return null;
            foreach (string drive in Directory.GetDirectories(cloud, "GoogleDrive-*"))
            {
                string mac = Path.Combine(
                    drive, "My Drive", "My Private", "Projects", "GenesisInteractive");
                if (Directory.Exists(mac))
                    return mac;
            }
            return null;
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
