using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using QwenTTS;
using Newtonsoft.Json;

namespace LiveTalk.API
{
    using Core;
    using Utils;
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

    /// <summary>
    /// <c>character.json</c>. The 2.0 layout is references only:
    /// <c>{ id, name, avatarId, voiceId, speechSampleRate, createdUtc }</c>.
    /// The pre-2.0 layout had <c>{ name, gender, pitch, speed, intro,
    /// voiceInstruct }</c> with the avatar and voice inline beside it; those
    /// fields are kept here so a legacy file still parses and can be told
    /// apart (<see cref="IsLegacy"/>). Newtonsoft only; not a Unity-serialized type.
    /// </summary>
    internal class CharacterFile
    {
        public string id;
        public string name;
        public string avatarId;
        public string voiceId;
        public int? speechSampleRate;
        public DateTime? createdUtc;
        public string version;

        // Pre-2.0 inline layout.
        public Gender? gender;
        public Pitch? pitch;
        public Speed? speed;
        public string intro;
        public string voiceInstruct;

        public const string FileName = "character.json";

        [JsonIgnore]
        public bool IsLegacy => string.IsNullOrEmpty(id) && string.IsNullOrEmpty(avatarId) && string.IsNullOrEmpty(voiceId);
    }

    /// <summary>
    /// A named composition of an <see cref="Avatar"/> (the face) and a
    /// <see cref="Voice"/> (the speaker). Creating one is instant — both halves
    /// already exist — and swapping the voice is a one-line write.
    ///
    /// <para><b>Identity.</b> <see cref="Id"/> is a GUID assigned by
    /// <see cref="LiveTalkAPI.CreateCharacter"/> and stable for the
    /// character's life, whatever its voice or name becomes.</para>
    ///
    /// <para><b>Storage.</b>
    /// <c>&lt;saveLocation&gt;/characters/&lt;Id&gt;/character.json</c> holding
    /// <c>{ id, name, avatarId, voiceId, speechSampleRate, createdUtc }</c>.
    /// The avatar and voice folders are referenced by id, never copied, so
    /// several characters share one avatar and deleting a character leaves
    /// both halves for others.</para>
    ///
    /// <para><b>Legacy.</b> Pre-2.0 characters lived in
    /// <c>&lt;saveLocation&gt;/&lt;id&gt;[.bundle]/</c> with the avatar and
    /// voice inline. <see cref="LiveTalkAPI.LoadCharacterAsyncFromId"/> still
    /// loads those in place (<see cref="IsLegacy"/>); they cannot have their
    /// voice replaced and their halves cannot be shared. Recreate them through
    /// the 2.0 API to migrate.</para>
    /// </summary>
    public sealed class Character
    {
        /// <summary>GUID assigned at creation; also the folder name under <c>characters/</c>. For a legacy character, its folder name.</summary>
        public string Id { get; private set; }

        /// <summary>Same as <see cref="Id"/>. Read-only; a character's id never changes.</summary>
        public string CharacterId => Id;

        /// <summary>Display name. Not part of the identity.</summary>
        public string Name { get; private set; }

        /// <summary>The face, or null for a voice-only character.</summary>
        public Avatar Avatar { get; private set; }

        /// <summary>The speaker. Never null once <see cref="IsDataLoaded"/>.</summary>
        public Voice Voice { get; private set; }

        /// <summary>
        /// The portrait: <see cref="Avatar"/>'s image once loaded, or just the
        /// image file after <see cref="LiveTalkAPI.LoadCharacterMetadataAsync"/>.
        /// </summary>
        public Texture2D Image { get; private set; }

        /// <summary>
        /// Sample rate for generated speech, or 0 for the TTS model's native rate.
        /// 16 kHz for a character with an animatable avatar because that is what
        /// the lip-sync stack consumes; native for a voice-only character, which
        /// matters when the clip is going to be a clone reference: the speaker
        /// encoder reads mel up to 12 kHz, so a 16 kHz round trip throws away the
        /// top of the band the speaker is identified by. Persisted in
        /// <c>character.json</c>.
        /// </summary>
        public int SpeechSampleRate { get; set; } = 16000;

        /// <summary>True once <see cref="Avatar"/> (if any) and <see cref="Voice"/> are loaded and the character can speak.</summary>
        public bool IsDataLoaded { get; private set; }

        /// <summary>True for a pre-2.0 inline folder. See the class remarks.</summary>
        public bool IsLegacy { get; private set; }

        /// <summary>When the character was created; <see cref="DateTime.MinValue"/> for a legacy file.</summary>
        public DateTime CreatedUtc { get; private set; }

        [Obsolete("Use Voice.Gender.")]
        public Gender Gender => Voice != null ? Voice.Gender : _legacyFile?.gender ?? default;
        [Obsolete("Use Voice.Pitch.")]
        public Pitch Pitch => Voice != null ? Voice.Pitch : _legacyFile?.pitch ?? default;
        [Obsolete("Use Voice.Speed.")]
        public Speed Speed => Voice != null ? Voice.Speed : _legacyFile?.speed ?? default;
        [Obsolete("Use Voice.SampleText.")]
        public string Intro => Voice != null ? Voice.SampleText : _legacyFile?.intro;
        [Obsolete("Use Voice.Instruct.")]
        public string VoiceInstruct => Voice != null ? Voice.Instruct : _legacyFile?.voiceInstruct;
        [Obsolete("Use Voice.SampleText.")]
        public string VoiceCloneRefText => Voice != null && Voice.Kind == VoiceKind.Cloned ? Voice.SampleText : null;
        [Obsolete("Use Voice.Sample.")]
        public AudioClip VoicePromptClip => Voice?.Sample;

        internal static string saveLocation
        {
            get => LiveTalkStorage.Root;
            set => LiveTalkStorage.Root = value;
        }

        /// <summary><c>characters/&lt;Id&gt;</c>, or the legacy inline folder.</summary>
        internal string CharacterFolder { get; private set; }

        /// <summary>Where the idle frames are (expression 0), or null without an animatable avatar.</summary>
        internal string IdleFramesFolder => Avatar != null && Avatar.CanAnimate ? Avatar.ExpressionFolder(0) : null;

        // Ids read from character.json before the halves are loaded
        // (metadata load), and the legacy file for the inline path.
        private string _avatarId;
        private string _voiceId;
        private CharacterFile _legacyFile;

        // CharacterPlayer for animation and playback
        private CharacterPlayer _characterPlayer;

        /// <summary>
        /// The player for this character, created on first access once the
        /// data is loaded. Null before that. <see cref="DestroyPlayer"/> tears
        /// it down; the next access makes a new one.
        /// </summary>
        public CharacterPlayer CharacterPlayer
        {
            get
            {
                if (_characterPlayer == null && IsDataLoaded)
                {
                    CreateCharacterPlayer();
                }
                return _characterPlayer;
            }
        }

        private Character(string id, string name)
        {
            Id = id;
            Name = name;
        }

        #region Create / persist

        /// <summary>
        /// Composes a loaded avatar and voice into a new character and writes
        /// its <c>character.json</c>. Synchronous: nothing is processed.
        /// </summary>
        internal static Character CreateNew(string name, Avatar avatar, Voice voice)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Character name cannot be null or empty.", nameof(name));
            if (voice == null)
                throw new ArgumentNullException(nameof(voice), "A character needs a voice. Design or clone one first.");
            if (voice.Loaded == null)
                throw new ArgumentException("The voice is not loaded.", nameof(voice));
            if (voice.IsLegacy)
                throw new ArgumentException(
                    "This voice lives inline in a pre-2.0 character folder and cannot be referenced by a new character. " +
                    "Clone or design a voice through the 2.0 API instead.", nameof(voice));
            if (avatar != null && avatar.IsLegacy)
                throw new ArgumentException(
                    "This avatar lives inline in a pre-2.0 character folder and cannot be referenced by a new character. " +
                    "Create it through CreateAvatarAsync instead.", nameof(avatar));

            var character = new Character(Guid.NewGuid().ToString("N"), name)
            {
                Avatar = avatar,
                Voice = voice,
                Image = avatar?.Image,
                CreatedUtc = DateTime.UtcNow,
                // Nothing lip-syncs a voice-only character, so hand back the
                // TTS model's own rate instead of the 16 kHz lip-sync rate.
                SpeechSampleRate = avatar != null && avatar.CanAnimate ? 16000 : 0,
            };
            character._avatarId = avatar?.Id;
            character._voiceId = voice.Id;
            character.CharacterFolder = LiveTalkStorage.CharacterFolder(character.Id);
            character.WriteFile();
            character.IsDataLoaded = true;

            Logger.Log($"[Character] Created character '{name}' ({character.Id}): avatar={avatar?.Id ?? "none"}, voice={voice.Id}");
            return character;
        }

        /// <summary>
        /// Points this character at a different voice. Rewrites
        /// <c>character.json</c>, and if a player exists, drops any speech
        /// still queued or in flight in the old voice so the next
        /// <see cref="CharacterPlayer.QueueSpeech"/> speaks in the new one.
        /// The avatar, id and name are untouched; nothing is reprocessed.
        /// </summary>
        /// <exception cref="ArgumentNullException">No voice.</exception>
        /// <exception cref="ArgumentException">The voice is not loaded, or lives inline in a pre-2.0 folder.</exception>
        /// <exception cref="InvalidOperationException">This is a pre-2.0 inline character; recreate it through <see cref="LiveTalkAPI.CreateCharacter"/>.</exception>
        public void ReplaceVoice(Voice voice)
        {
            if (voice == null)
                throw new ArgumentNullException(nameof(voice));
            if (voice.Loaded == null)
                throw new ArgumentException("The voice is not loaded.", nameof(voice));
            if (voice.IsLegacy)
                throw new ArgumentException(
                    "This voice lives inline in a pre-2.0 character folder and cannot be referenced.", nameof(voice));
            if (IsLegacy)
                throw new InvalidOperationException(
                    $"Character '{Name}' is a pre-2.0 inline character; its voice cannot be replaced in place. " +
                    "Create a new character from its avatar and the new voice instead.");
            if (!IsDataLoaded)
                throw new InvalidOperationException(
                    $"Character '{Name}' is not loaded. Load it fully before replacing its voice.");

            var previous = Voice;
            Voice = voice;
            _voiceId = voice.Id;
            WriteFile();

            // Anything generated in the old voice must not play as the new one.
            if (_characterPlayer != null)
                _characterPlayer.OnVoiceReplaced();

            Logger.Log($"[Character] '{Name}' voice replaced: {previous?.Id ?? "none"} → {voice.Id}");
        }

        /// <summary>Writes <c>character.json</c> for a 2.0 character. Synchronous; the file is a few hundred bytes.</summary>
        private void WriteFile()
        {
            var file = new CharacterFile
            {
                id = Id,
                name = Name,
                avatarId = _avatarId,
                voiceId = _voiceId,
                speechSampleRate = SpeechSampleRate,
                createdUtc = CreatedUtc,
                version = "2.0",
            };
            Directory.CreateDirectory(CharacterFolder);
            File.WriteAllText(Path.Combine(CharacterFolder, CharacterFile.FileName),
                JsonConvert.SerializeObject(file, Formatting.Indented,
                    new JsonSerializerSettings { NullValueHandling = NullValueHandling.Ignore }));
        }

        #endregion

        #region Player

        /// <summary>
        /// Creates and initializes the CharacterPlayer for this character
        /// </summary>
        private void CreateCharacterPlayer()
        {
            if (_characterPlayer != null || !IsDataLoaded)
                return;
            
            // Create GameObject for CharacterPlayer
            var playerObject = new GameObject($"CharacterPlayer_{Name}");
            playerObject.transform.SetParent(CharacterPlayer.ParentTransform);
            
            // Add CharacterPlayer component
            _characterPlayer = playerObject.AddComponent<CharacterPlayer>();
            
            // Assign this character to the player
            _characterPlayer.AssignCharacter(this);
            
            Logger.Log($"[Character] Created CharacterPlayer for {Name}");
        }

        /// <summary>
        /// Stops and destroys the <see cref="CharacterPlayer"/> GameObject, if
        /// one was created. Safe to call when there is none, or when the host
        /// already destroyed it. The next <see cref="CharacterPlayer"/> access
        /// creates a fresh one. Hosts that rebuild characters should call this
        /// so idle players and their AudioSources do not accumulate.
        /// </summary>
        public void DestroyPlayer()
        {
            var player = _characterPlayer;
            _characterPlayer = null;
            if (player == null) // Unity null: also true for an already-destroyed component
                return;

            player.Stop();
            var go = player.gameObject;
            if (Application.isPlaying)
                UnityEngine.Object.Destroy(go);
            else
                UnityEngine.Object.DestroyImmediate(go);
            Logger.Log($"[Character] Destroyed CharacterPlayer for {Name}");
        }

        #endregion

        #region Load

        /// <remarks>
        /// Exactly one of <paramref name="onComplete"/> / <paramref name="onError"/>
        /// fires. A character whose avatar or voice is missing or fails to load
        /// is reported through <paramref name="onError"/>, naming the missing
        /// half, rather than handed back half-loaded.
        /// </remarks>
        public static IEnumerator LoadCharacterAsyncFromPath(
            string characterPath,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            return TaskYield.Guard(LoadCharacterFromPathCore(characterPath, onComplete), onError,
                "Character.LoadCharacterAsyncFromPath");
        }

        private static IEnumerator LoadCharacterFromPathCore(string characterPath, Action<Character> onComplete)
        {
            if (string.IsNullOrEmpty(characterPath))
                throw new ArgumentException("Character path cannot be null or empty.");
            Logger.Log($"[Character] Loading character from: {characterPath}");

            var start = System.Diagnostics.Stopwatch.StartNew();
            Character character = null;
            yield return ReadFileCore(characterPath, c => character = c);
            yield return character.LoadData();

            Logger.LogVerbose($"[Character] Character '{character.Name}' loaded in {start.Elapsed.TotalMilliseconds:F0}ms");
            onComplete?.Invoke(character);
        }

        public static IEnumerator LoadCharacterAsyncFromId(
            string characterId,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            if (string.IsNullOrEmpty(characterId))
            {
                onError?.Invoke(new ArgumentException("Character ID cannot be null or empty."));
                yield break;
            }

            string characterPath = GetCharacterPath(characterId);
            if (characterPath == null)
            {
                onError?.Invoke(new DirectoryNotFoundException(
                    $"Character not found: {characterId} (checked characters/{characterId}, and the legacy {characterId}.bundle / {characterId} folders)"));
                yield break;
            }

            yield return LoadCharacterAsyncFromPath(characterPath, onComplete, onError);
        }

        /// <summary>
        /// Load only character metadata (name + image) without expressions/voice by ID.
        /// This is a lightweight load for thumbnails and lists.
        /// </summary>
        public static IEnumerator LoadCharacterMetadataAsync(
            string characterId,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            if (string.IsNullOrEmpty(characterId))
            {
                onError?.Invoke(new ArgumentException("Character ID cannot be null or empty."));
                yield break;
            }

            string characterPath = GetCharacterPath(characterId);
            if (characterPath == null)
            {
                onError?.Invoke(new DirectoryNotFoundException($"Character not found: {characterId}"));
                yield break;
            }

            yield return LoadCharacterMetadataFromPathAsync(characterPath, onComplete, onError);
        }

        /// <summary>
        /// Load only character metadata (name + image) without expressions/voice from path.
        /// This is a lightweight load for thumbnails and lists.
        /// </summary>
        public static IEnumerator LoadCharacterMetadataFromPathAsync(
            string characterPath,
            Action<Character> onComplete,
            Action<Exception> onError)
        {
            return TaskYield.Guard(LoadCharacterMetadataFromPathCore(characterPath, onComplete), onError,
                "Character.LoadCharacterMetadataFromPathAsync");
        }

        private static IEnumerator LoadCharacterMetadataFromPathCore(string characterPath, Action<Character> onComplete)
        {
            if (string.IsNullOrEmpty(characterPath))
                throw new ArgumentException("Character path cannot be null or empty.");

            Character character = null;
            yield return ReadFileCore(characterPath, c => character = c);

            // Image only — no frames, no voice.
            string imagePath = character.IsLegacy
                ? Path.Combine(characterPath, Avatar.ImageFileName)
                : string.IsNullOrEmpty(character._avatarId)
                    ? null
                    : Path.Combine(LiveTalkStorage.AvatarFolder(character._avatarId), Avatar.ImageFileName);
            if (imagePath != null && File.Exists(imagePath))
            {
                byte[] imageBytes = null;
                yield return TaskYield.Wait(File.ReadAllBytesAsync(imagePath), b => imageBytes = b,
                    $"Character.LoadMetadata read {imagePath}");
                var texture = new Texture2D(2, 2);
                if (texture.LoadImage(imageBytes))
                    character.Image = texture;
                else
                    UnityEngine.Object.DestroyImmediate(texture);
            }

            Logger.Log($"[Character] Loaded metadata for {character.Name}");
            onComplete?.Invoke(character);
        }

        /// <summary>
        /// Reads <c>character.json</c> into an unloaded <see cref="Character"/>:
        /// id, name, folder, referenced ids (or the legacy file). Throws when
        /// the file is missing or unreadable.
        /// </summary>
        private static IEnumerator ReadFileCore(string characterFolder, Action<Character> onRead)
        {
            string configPath = Path.Combine(characterFolder, CharacterFile.FileName);
            if (!File.Exists(configPath))
                throw new FileNotFoundException($"Character config file not found: {configPath}", configPath);

            string json = null;
            yield return TaskYield.Wait(File.ReadAllTextAsync(configPath), t => json = t,
                $"Character.ReadFile {configPath}");

            CharacterFile file;
            try
            {
                file = JsonConvert.DeserializeObject<CharacterFile>(json)
                       ?? throw new InvalidDataException("empty document");
            }
            catch (Exception ex)
            {
                throw new InvalidDataException($"Failed to parse {configPath}: {ex.Message}", ex);
            }

            string folderId = Path.GetFileNameWithoutExtension(characterFolder);
            string id = string.IsNullOrEmpty(file.id) ? folderId : file.id;
            var character = new Character(id, string.IsNullOrEmpty(file.name) ? folderId : file.name)
            {
                CharacterFolder = characterFolder,
                IsLegacy = file.IsLegacy,
                CreatedUtc = file.createdUtc ?? DateTime.MinValue,
            };
            if (file.IsLegacy)
            {
                character._legacyFile = file;
                Logger.Log($"[Character] '{character.Name}' is a pre-2.0 inline character folder ({characterFolder}); " +
                           "loading in place. Recreate it through CreateCharacter to migrate.");
            }
            else
            {
                character._avatarId = file.avatarId;
                character._voiceId = file.voiceId;
                if (file.speechSampleRate.HasValue)
                    character.SpeechSampleRate = file.speechSampleRate.Value;
                if (string.IsNullOrEmpty(file.voiceId))
                    throw new InvalidDataException($"{configPath} references no voice; the character cannot speak.");
            }
            onRead(character);
        }

        /// <summary>
        /// Loads the avatar (if any) and the voice this character references,
        /// or the inline halves of a legacy folder. Throws — so the guarded
        /// caller's onError fires — naming whichever half is missing, rather
        /// than marking a character loaded without it.
        /// </summary>
        internal IEnumerator LoadData()
        {
            if (IsDataLoaded)
                yield break;
            if (string.IsNullOrEmpty(CharacterFolder))
                throw new InvalidOperationException($"Character '{Name}' has no folder to load from.");

            Logger.LogVerbose($"[Character] Loading character data for {Name}");

            if (IsLegacy)
            {
                yield return LoadLegacyInline();
            }
            else
            {
                yield return LoadReferenced();
            }

            if (Voice == null || Voice.Loaded == null)
            {
                // Both loaders throw on every failure they can see, so this is
                // the belt to that brace: IsDataLoaded must never be true for
                // a character that cannot speak.
                throw new InvalidOperationException(
                    $"Voice for character '{Name}' did not load; the character is not usable. See the earlier error for the cause.");
            }

            Image = Avatar?.Image ?? Image;
            IsDataLoaded = true;
            Logger.LogVerbose($"[Character] Character data loaded successfully for {Name}");
        }

        private IEnumerator LoadReferenced()
        {
            // Check both halves before loading either, so one error can name
            // everything that is missing.
            string avatarFolder = string.IsNullOrEmpty(_avatarId) ? null : LiveTalkStorage.AvatarFolder(_avatarId);
            string voiceFolder = LiveTalkStorage.VoiceFolder(_voiceId);
            string missing = null;
            if (avatarFolder != null && !Directory.Exists(avatarFolder))
                missing = $"avatar {_avatarId} (expected at {avatarFolder})";
            if (!Directory.Exists(voiceFolder))
                missing = (missing == null ? "" : missing + " and ") + $"voice {_voiceId} (expected at {voiceFolder})";
            if (missing != null)
            {
                throw new DirectoryNotFoundException(
                    $"Character '{Name}' ({Id}) references {missing}, which is missing. " +
                    "It was deleted, or the save location moved without it.");
            }

            if (avatarFolder != null)
            {
                Avatar avatar = null;
                yield return Avatar.LoadCore(avatarFolder, _avatarId, modeHint: null, isLegacy: false, a => avatar = a);
                Avatar = avatar;
            }

            Voice voice = null;
            yield return TaskYield.Wait(Voice.LoadAsync(voiceFolder, _voiceId, isLegacy: false, fallbackMeta: null),
                v => voice = v, $"Character.LoadVoice {voiceFolder}");
            Voice = voice;
        }

        private IEnumerator LoadLegacyInline()
        {
            // The avatar half is inline: image.png + drivingFrames/ beside the
            // json. A voice-only legacy character has neither.
            bool hasImage = File.Exists(Path.Combine(CharacterFolder, Avatar.ImageFileName));
            bool hasFrames = Directory.Exists(Path.Combine(CharacterFolder, Avatar.DrivingFramesFolderName));
            if (hasImage || hasFrames)
            {
                Avatar avatar = null;
                yield return Avatar.LoadCore(CharacterFolder, Id, modeHint: null, isLegacy: true, a => avatar = a);
                Avatar = avatar;
            }

            string voiceFolder = Path.Combine(CharacterFolder, "voice");
            if (!Directory.Exists(voiceFolder))
            {
                throw new DirectoryNotFoundException(
                    $"No voice folder for character '{Name}': {voiceFolder}. " +
                    "The character was never given a voice, or its creation failed part-way.");
            }

            // Voice.LoadAsync corrects the kind (and the text) from the engine
            // manifest when the folder turns out to hold a clone.
            var fallback = new VoiceMeta
            {
                id = Id,
                kind = VoiceKind.Designed,
                sampleText = _legacyFile?.intro,
                gender = _legacyFile?.gender ?? default,
                pitch = _legacyFile?.pitch ?? default,
                speed = _legacyFile?.speed ?? default,
                instruct = _legacyFile?.voiceInstruct,
            };
            Voice voice = null;
            yield return TaskYield.Wait(Voice.LoadAsync(voiceFolder, Id, isLegacy: true, fallbackMeta: fallback),
                v => voice = v, $"Character.LoadVoice {voiceFolder}");
            Voice = voice;

            // Legacy files did not persist the rate; apply the creation-time rule.
            SpeechSampleRate = Avatar != null && Avatar.CanAnimate ? 16000 : 0;
        }

        /// <summary>
        /// Get the full path to a character by ID: the 2.0 <c>characters/</c>
        /// folder first, then the legacy <c>.bundle</c> and plain folders.
        /// </summary>
        /// <returns>The full path to the character folder/bundle, or null if not found</returns>
        internal static string GetCharacterPath(string characterId)
        {
            if (!LiveTalkStorage.HasRoot || string.IsNullOrEmpty(characterId))
                return null;

            string root = LiveTalkStorage.Root;
            string[] candidates =
            {
                Path.Combine(root, LiveTalkStorage.CharactersFolderName, characterId),
                Path.Combine(root, $"{characterId}.bundle"),
                Path.Combine(root, characterId),
            };
            foreach (var candidate in candidates)
            {
                if (Directory.Exists(candidate) && File.Exists(Path.Combine(candidate, CharacterFile.FileName)))
                    return candidate;
            }
            return null;
        }

        #endregion

        #region Speak

        /// <summary>
        /// Generate speech asynchronously using coroutines with optional caching.
        /// Speech audio is automatically cached using the global Cache if enabled.
        /// Provides two callbacks: one when audio is ready, another when animation completes.
        /// Uses queuing to prevent parallel model usage.
        /// </summary>
        /// <param name="text">Text to speak</param>
        /// <param name="expressionIndex">Expression to use, -1 for voice only</param>
        /// <param name="onAudioReady">Callback when audio generation is complete. Called with (FrameStream, AudioClip). 
        /// FrameStream will receive frames as they're generated. Caller can schedule next SpeakAsync here.</param>
        /// <param name="onAnimationComplete">Callback when animation generation is complete. Called with the final FrameStream.
        /// For voice-only mode (expressionIndex=-1), this is called immediately after onAudioReady.</param>
        /// <param name="onError">Callback when an error occurs</param>
        /// <param name="onSpeechChunk">
        /// Optional. Receives speech as it is generated, rather than only when
        /// the line is finished: generation runs slightly faster than playback,
        /// so the first chunk arrives in about a second. Each call carries only
        /// samples not reported before, so appending them in order reproduces
        /// the utterance. Delivered on the main thread.
        ///
        /// Ignored for a cache hit, since there is nothing to stream — the
        /// whole clip is already on disk and arrives via onAudioReady.
        /// </param>
        /// <returns>Coroutine for audio generation</returns>
        /// <remarks>
        /// Audio is cached on <c>(Voice.Id, text)</c> and frames on
        /// <c>(Voice.Id, text, Avatar.Id, expressionIndex)</c>, so a replaced
        /// voice never replays old takes and the same line at two expressions
        /// never shares frames.
        ///
        /// Every failure reaches <paramref name="onError"/>: a faulted speech
        /// synthesis, a lip-sync model that failed to load, a driving-frame
        /// cache that could not be read. Failures after audio was handed to
        /// <paramref name="onAudioReady"/> still fire <paramref name="onError"/>,
        /// and the <see cref="FrameStream"/> is marked finished with its
        /// <see cref="FrameStream.Error"/> set, so a consumer draining it exits.
        /// <paramref name="onAnimationComplete"/> is not called for a failed
        /// animation.
        /// </remarks>
        public IEnumerator SpeakAsync(
            string text, 
            int expressionIndex = 0,
            Action<FrameStream, AudioClip> onAudioReady = null,
            Action<FrameStream> onAnimationComplete = null,
            Action<Exception> onError = null,
            Action<float[], int> onSpeechChunk = null)
        {
            return TaskYield.Guard(
                SpeakCore(text, expressionIndex, onAudioReady, onAnimationComplete, onError, onSpeechChunk),
                onError,
                "Character.SpeakAsync");
        }

        private IEnumerator SpeakCore(
            string text,
            int expressionIndex,
            Action<FrameStream, AudioClip> onAudioReady,
            Action<FrameStream> onAnimationComplete,
            Action<Exception> onError,
            Action<float[], int> onSpeechChunk)
        {
            var start = System.Diagnostics.Stopwatch.StartNew();
            if (!IsDataLoaded)
            {
                onError?.Invoke(new InvalidOperationException(
                    "Character data not loaded. Use LiveTalkAPI.LoadCharacterAsyncFromId() or CreateCharacter() first, " +
                    "and check that call's onError — a character whose load failed stays unloaded."));
                yield break;
            }

            if (string.IsNullOrEmpty(text))
            {
                onError?.Invoke(new ArgumentException("Text cannot be null or empty."));
                yield break;
            }

            if (expressionIndex != -1)
            {
                if (Avatar == null || !Avatar.CanAnimate)
                {
                    onError?.Invoke(new ArgumentException(
                        $"Character '{Name}' has no animatable avatar; use expressionIndex -1 for voice only."));
                    yield break;
                }
                if (!Avatar.LoadedExpressions.ContainsKey(expressionIndex))
                {
                    onError?.Invoke(new ArgumentException($"Expression index {expressionIndex} not available. Available expressions: {string.Join(", ", Avatar.LoadedExpressions.Keys)}"));
                    yield break;
                }
            }

            var voice = Voice?.Loaded;
            if (voice == null)
            {
                onError?.Invoke(new InvalidOperationException(
                    "Character voice not loaded — the voice folder is missing or the voice failed to design/clone/load; " +
                    "see the earlier error from character creation or load."));
                yield break;
            }

            var liveTalkAPI = LiveTalkAPI.Instance;
            if (liveTalkAPI == null)
            {
                onError?.Invoke(new InvalidOperationException("LiveTalkAPI not initialized. Call LiveTalkAPI.Initialize() first."));
                yield break;
            }

            Logger.LogVerbose($"[Character] {Name} speaking async: \"{text}\" with expression {expressionIndex}");

            AudioClip audioClip = null;
            string cacheKey = null;
            string framesCacheKey = null;
            bool audioFromCache = false;

            // Check audio cache first
            if (LiveTalkCache.IsEnabled && !string.IsNullOrEmpty(Voice.Id))
            {
                cacheKey = HashUtils.GenerateSpeechCacheKey(Voice.Id, text);
                if (expressionIndex != -1)
                    framesCacheKey = HashUtils.GenerateFramesCacheKey(Voice.Id, text, Avatar.Id, expressionIndex);
                var (exists, cachedPath) = LiveTalkCache.CheckExists(cacheKey);
                
                if (exists)
                {
                    Logger.LogVerbose($"[Character] Loading cached audio for: {text[..Math.Min(30, text.Length)]}...");
                    var loadTask = AudioFileIO.LoadClipAsync(cachedPath);
                    yield return new WaitUntil(() => loadTask.IsCompleted);

                    // A cache hit that cannot be read is not fatal — the line
                    // is regenerated below — but it is not silent either.
                    if (loadTask.IsFaulted)
                    {
                        Logger.LogWarning($"[Character] Cached audio unreadable, regenerating: {cachedPath}: " +
                            loadTask.Exception?.GetBaseException().Message);
                    }
                    else if (loadTask.Result != null)
                    {
                        audioClip = loadTask.Result;
                        audioFromCache = true;
                    }
                }
            }

            // Generate new audio if not cached (with queuing)
            if (audioClip == null)
            {
                // Acquire voice queue lock. The lease is released in the
                // finally below on every exit: success, a fault rethrown by
                // the bridge, or the host stopping this coroutine.
                yield return TaskYield.Wait(liveTalkAPI.VoiceQueue.AcquireAsync(), "Character.VoiceQueue.Acquire");

                try
                {
                    var options = new SpeechOptions { SampleRate = SpeechSampleRate };

                    // Progress<T> captures the SynchronizationContext it is
                    // built on. This coroutine runs on the main thread, so the
                    // engine's worker-thread reports arrive back here rather
                    // than on the thread that generated them — which matters,
                    // because a host will want to hand these to an AudioSource.
                    var chunkRelay = onSpeechChunk == null
                        ? null
                        : new Progress<SpeechChunk>(c => onSpeechChunk(c.Pcm, c.SampleRate));

                    var audioTask = chunkRelay == null
                        ? voice.SpeakAsync(text, options)
                        : voice.SpeakStreamAsync(text, chunkRelay, options);

                    // A faulted synthesis rethrows here, unwinds through the
                    // finally (releasing the lease) and reaches onError via
                    // the Guard in SpeakAsync.
                    SpeechResult speech = default;
                    yield return TaskYield.Wait(audioTask, r => speech = r, "Character.SpeakAsync (TTS)");

                    // ToAudioClip has to happen here: it is a main-thread API.
                    audioClip = speech.ToAudioClip($"{Name}_speech");
                }
                finally
                {
                    // Release voice queue lock
                    liveTalkAPI.VoiceQueue.Release();
                }
                
                // Save audio to cache (fire and forget)
                if (LiveTalkCache.IsEnabled && !string.IsNullOrEmpty(cacheKey) && audioClip != null)
                {
                    string cachePath = LiveTalkCache.GetFilePath(cacheKey);
                    if (!string.IsNullOrEmpty(cachePath))
                    {
                        var saveTask = AudioFileIO.SaveClipAsync(audioClip, cachePath);
                        _ = saveTask.ContinueWith(t => 
                        {
                            if (t.IsFaulted)
                                Logger.LogWarning($"[Character] Failed to save audio to cache: {t.Exception?.InnerException?.Message}");
                            else
                                Logger.LogVerbose($"[Character] Saved audio to cache: {cacheKey}");
                        });
                    }
                }
            }

            if (audioClip == null)
            {
                onError?.Invoke(new InvalidOperationException(
                    "Generated audio clip is null — the TTS engine returned no audio for this line; see the earlier error."));
                yield break;
            }

            var outputStream = new FrameStream(0);
            
            // For voice-only mode, both callbacks immediately
            if (expressionIndex == -1)
            {
                onAudioReady?.Invoke(outputStream, audioClip);
                onAnimationComplete?.Invoke(outputStream);
                var stopLocal = start.Elapsed;
                Logger.Log($"[Character] Speaking completed for {Name} in {stopLocal.TotalMilliseconds}ms{(audioFromCache ? " (cached)" : "")}");
                yield break;
            }

            // Check for cached animation frames
            if (LiveTalkCache.IsEnabled && !string.IsNullOrEmpty(framesCacheKey))
            {
                var (framesExist, framesFolder, frameCount) = LiveTalkCache.CheckFramesCacheExists(framesCacheKey);
                
                if (framesExist && frameCount > 0)
                {
                    Logger.LogVerbose($"[Character] Loading {frameCount} cached animation frames for: {text[..Math.Min(30, text.Length)]}...");
                    
                    // Load frames from cache into output stream
                    outputStream = new FrameStream(frameCount);
                    
                    // Audio ready callback
                    onAudioReady?.Invoke(outputStream, audioClip);
                    
                    // Load frames and call animation complete when done. Guarded
                    // so a failed read finishes the stream and reaches onError
                    // rather than dying inside Unity's coroutine scheduler.
                    var cachedStream = outputStream;
                    liveTalkAPI.Controller.StartCoroutine(TaskYield.Guard(
                        LoadFramesFromCacheWithCallback(framesFolder, frameCount, cachedStream, onAnimationComplete),
                        ex => { cachedStream.Fail(ex); onError?.Invoke(ex); },
                        "Character.LoadFramesFromCache"));
                    
                    var stopCached = start.Elapsed;
                    Logger.Log($"[Character] Audio ready for {Name} in {stopCached.TotalMilliseconds}ms (audio+frames cached, loading...)");
                    yield break;
                }
            }

            // Audio ready - callback immediately, animation will be generated in background
            var expressionData = Avatar.LoadedExpressions[expressionIndex];
            outputStream = new FrameStream(0); // Will be updated with actual count when generation starts
            
            // Audio ready callback
            onAudioReady?.Invoke(outputStream, audioClip);
            var stopAudio = start.Elapsed;
            Logger.Log($"[Character] Audio ready for {Name} in {stopAudio.TotalMilliseconds}ms{(audioFromCache ? " (cached)" : "")}, animation pending...");

            // Start animation generation in background with queuing. The
            // Guard is what turns a fault inside the animation — a lip-sync
            // model that failed to load, most often — into a finished stream
            // plus an onError call, instead of a consumer that waits forever
            // and a MuseTalk lease that is never given back.
            var animationStream = outputStream;
            liveTalkAPI.Controller.StartCoroutine(TaskYield.Guard(
                GenerateAnimationWithQueue(liveTalkAPI, expressionData.Data, audioClip, animationStream, framesCacheKey, onAnimationComplete),
                ex => { animationStream.Fail(ex); onError?.Invoke(ex); },
                "Character.GenerateAnimation"));
        }

        /// <summary>
        /// Generate animation frames with queuing to prevent parallel MuseTalk usage.
        /// </summary>
        private static IEnumerator GenerateAnimationWithQueue(
            LiveTalkAPI liveTalkAPI,
            AvatarData avatarData,
            AudioClip audioClip,
            FrameStream outputStream,
            string framesCacheKey,
            Action<FrameStream> onAnimationComplete)
        {
            // Acquire MuseTalk queue lock. Released in the finally below on
            // every exit — success, fault, or the coroutine being disposed —
            // so a failed animation can never wedge the next one on Acquire.
            yield return TaskYield.Wait(liveTalkAPI.MuseTalkQueue.AcquireAsync(), "Character.MuseTalkQueue.Acquire");

            string framesFolder = null;
            bool completed = false;
            try
            {
                // Generate talking head using MuseTalk with preloaded data
                var generatedStream = liveTalkAPI.GenerateTalkingHeadWithPreloadedData(
                    avatarData,
                    audioClip
                );
                outputStream.TotalExpectedFrames = generatedStream.TotalExpectedFrames;

                // Forward frames from generated stream to output stream
                // If caching is enabled, also save frames
                if (LiveTalkCache.IsEnabled && !string.IsNullOrEmpty(framesCacheKey))
                {
                    framesFolder = LiveTalkCache.CreateFramesCacheFolder(framesCacheKey);
                }

                int frameIndex = 0;
                while (generatedStream.HasMoreFrames)
                {
                    var awaiter = generatedStream.WaitForNext();
                    yield return awaiter;

                    if (awaiter.Texture != null)
                    {
                        // Forward frame to output stream
                        outputStream.Queue.Enqueue(awaiter.Texture);

                        // Cache frame if enabled
                        if (!string.IsNullOrEmpty(framesFolder))
                        {
                            byte[] pngData = awaiter.Texture.EncodeToPNG();
                            int currentIndex = frameIndex;
                            _ = Task.Run(() =>
                            {
                                try
                                {
                                    string framePath = Path.Combine(framesFolder, $"frame_{currentIndex:D6}.png");
                                    File.WriteAllBytes(framePath, pngData);
                                }
                                catch (Exception ex)
                                {
                                    Logger.LogWarning($"[Character] Failed to cache frame {currentIndex}: {ex.Message}");
                                }
                            });
                        }

                        frameIndex++;
                    }
                }

                // The producer finishing early because it faulted must not
                // become a short animation that looks complete — and its
                // partial frames folder must not become a cache hit next time.
                if (generatedStream.Error != null)
                {
                    throw new InvalidOperationException(
                        $"Lip-sync animation failed after {frameIndex} frame(s): {generatedStream.Error.Message}",
                        generatedStream.Error);
                }

                outputStream.TotalExpectedFrames = frameIndex;
                outputStream.Finished = true;
                completed = true;
                Logger.LogVerbose($"[Character] Animation generation completed: {frameIndex} frames");
                
                // Animation complete callback
                onAnimationComplete?.Invoke(outputStream);
            }
            finally
            {
                // Consumers draining outputStream exit on every path; the
                // Guard that started this coroutine records the error on it.
                outputStream.Finished = true;

                // A run that faulted or was stopped part-way leaves a short
                // frames folder that the next SpeakAsync would take as a hit.
                if (!completed && framesFolder != null)
                {
                    LiveTalkCache.DeleteFramesCache(framesCacheKey);
                }

                // Release MuseTalk queue lock
                liveTalkAPI.MuseTalkQueue.Release();
            }
        }

        /// <summary>
        /// Load cached animation frames from disk into a FrameStream with completion callback.
        /// </summary>
        private static IEnumerator LoadFramesFromCacheWithCallback(
            string framesFolder, 
            int frameCount, 
            FrameStream outputStream,
            Action<FrameStream> onAnimationComplete)
        {
            yield return LoadFramesFromCache(framesFolder, frameCount, outputStream);
            onAnimationComplete?.Invoke(outputStream);
        }

        /// <summary>
        /// Load cached animation frames from disk into a FrameStream.
        /// </summary>
        /// <param name="framesFolder">Path to the folder containing cached frames</param>
        /// <param name="frameCount">Number of frames to load</param>
        /// <param name="outputStream">The output stream to populate with frames</param>
        private static IEnumerator LoadFramesFromCache(string framesFolder, int frameCount, FrameStream outputStream)
        {
            outputStream.TotalExpectedFrames = frameCount;

            try
            {
                for (int i = 0; i < frameCount; i++)
                {
                    string framePath = Path.Combine(framesFolder, $"frame_{i:D6}.png");

                    if (!File.Exists(framePath))
                    {
                        Logger.LogWarning($"[Character] Cached frame not found: {framePath}");
                        continue;
                    }

                    // Load frame from disk. A single unreadable cached frame is
                    // skipped (the fault is observed, logged, and the rest of
                    // the clip still plays); the cache entry is best-effort.
                    var loadTask = Task.Run(() => File.ReadAllBytes(framePath));
                    yield return new WaitUntil(() => loadTask.IsCompleted);

                    if (loadTask.IsFaulted)
                    {
                        Logger.LogWarning($"[Character] Failed to load cached frame {i}: {loadTask.Exception?.GetBaseException().Message}");
                        continue;
                    }

                    // Create texture from bytes
                    var texture = new Texture2D(2, 2);
                    if (texture.LoadImage(loadTask.Result))
                    {
                        texture.name = $"cached_frame_{i}";
                        outputStream.Queue.Enqueue(texture);
                    }
                    else
                    {
                        UnityEngine.Object.DestroyImmediate(texture);
                        Logger.LogWarning($"[Character] Failed to decode cached frame {i}");
                    }
                }

                Logger.LogVerbose($"[Character] Loaded {frameCount} frames from cache");
            }
            finally
            {
                outputStream.Finished = true;
            }
        }

        #endregion
    }
}
