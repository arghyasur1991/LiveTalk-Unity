using System;
using System.IO;

namespace LiveTalk.API
{
    using Utils;

    /// <summary>
    /// Where LiveTalk keeps what it makes. Everything hangs off
    /// <see cref="LiveTalkAPI.CharacterSaveLocation"/>:
    ///
    /// <code>
    /// &lt;saveLocation&gt;/
    ///   avatars/&lt;avatarId&gt;/      image.png, avatar.json, drivingFrames/expression-N/…
    ///   voices/&lt;voiceId&gt;/        voice.json, clone_prompt.bin, reference.wav, sample.wav, voice.meta.json
    ///   characters/&lt;characterId&gt;/ character.json   (references an avatar id and a voice id)
    ///   &lt;legacyId&gt;[.bundle]/      character.json + image.png + drivingFrames/ + voice/   (pre-2.0 inline layout, read-only)
    /// </code>
    ///
    /// Avatars and voices are content-addressed or unique by construction, so
    /// their folders double as the cache: asking for the same avatar again
    /// loads the folder instead of rebuilding it.
    /// </summary>
    internal static class LiveTalkStorage
    {
        public const string AvatarsFolderName = "avatars";
        public const string VoicesFolderName = "voices";
        public const string CharactersFolderName = "characters";

        /// <summary>Suffix of a folder still being written. Never matches an id lookup.</summary>
        private const string StagingSuffix = ".partial-";

        internal static string Root { get; set; }

        public static bool HasRoot => !string.IsNullOrEmpty(Root);

        public static string AvatarsRoot => Combine(AvatarsFolderName);
        public static string VoicesRoot => Combine(VoicesFolderName);
        public static string CharactersRoot => Combine(CharactersFolderName);

        public static string AvatarFolder(string avatarId) => Combine(AvatarsFolderName, avatarId);
        public static string VoiceFolder(string voiceId) => Combine(VoicesFolderName, voiceId);
        public static string CharacterFolder(string characterId) => Combine(CharactersFolderName, characterId);

        private static string Combine(params string[] parts)
        {
            if (!HasRoot)
                throw new InvalidOperationException(
                    "LiveTalkAPI has no save location. Call LiveTalkAPI.Initialize() first.");
            foreach (var part in parts)
            {
                if (string.IsNullOrEmpty(part))
                    throw new ArgumentException("Storage id cannot be null or empty.");
                if (part.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || part.Contains(".."))
                    throw new ArgumentException($"Storage id is not a valid folder name: '{part}'.");
            }
            string path = Root;
            foreach (var part in parts)
                path = Path.Combine(path, part);
            return path;
        }

        /// <summary>
        /// Folder to write a new entry into. It sits beside the final folder
        /// with a suffix that no id lookup matches, so a crash or a fault
        /// mid-write cannot leave something that looks complete.
        /// </summary>
        public static string BeginStaging(string finalFolder)
        {
            string staging = finalFolder + StagingSuffix + Guid.NewGuid().ToString("N").Substring(0, 8);
            Directory.CreateDirectory(staging);
            return staging;
        }

        /// <summary>
        /// Moves a finished staging folder into place. Returns false — and
        /// deletes the staging folder — if the final folder appeared meanwhile
        /// (another create of the same content won the race); the caller
        /// should then load the winner.
        /// </summary>
        public static bool CommitStaging(string staging, string finalFolder)
        {
            if (Directory.Exists(finalFolder))
            {
                Logger.LogVerbose($"[Storage] {finalFolder} already exists; discarding duplicate {staging}");
                DeleteFolder(staging);
                return false;
            }
            Directory.CreateDirectory(Path.GetDirectoryName(finalFolder));
            Directory.Move(staging, finalFolder);
            return true;
        }

        /// <summary>Removes a folder if it exists; never throws.</summary>
        public static void DeleteFolder(string folder)
        {
            if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
                return;
            try
            {
                Directory.Delete(folder, true);
            }
            catch (Exception ex)
            {
                Logger.LogWarning($"[Storage] Could not delete {folder}: {ex.Message}");
            }
        }

        /// <summary>
        /// Deletes staging folders left behind by a crash. Called on
        /// initialize; cheap, and it is what keeps the layout honest.
        /// </summary>
        public static void SweepStaging()
        {
            if (!HasRoot || !Directory.Exists(Root))
                return;
            foreach (var root in new[] { AvatarsRoot, VoicesRoot })
            {
                if (!Directory.Exists(root))
                    continue;
                foreach (var dir in Directory.GetDirectories(root))
                {
                    if (Path.GetFileName(dir).Contains(StagingSuffix))
                    {
                        Logger.Log($"[Storage] Removing unfinished folder {dir}");
                        DeleteFolder(dir);
                    }
                }
            }
        }
    }
}
