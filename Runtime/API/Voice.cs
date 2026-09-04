using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using QwenTTS;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace LiveTalk.API
{
    using Utils;

    /// <summary>How a <see cref="Voice"/> came to be.</summary>
    public enum VoiceKind
    {
        /// <summary>
        /// Sampled by the VoiceDesign checkpoint from a description. Every
        /// design call draws a new speaker, so every designed voice is a
        /// distinct object with its own id.
        /// </summary>
        Designed,
        /// <summary>
        /// Cloned from a reference recording (in-context when a transcript is
        /// supplied). Deterministic: the same reference and transcript always
        /// produce the same voice id.
        /// </summary>
        Cloned,
    }

    /// <summary>
    /// <c>voice.meta.json</c>: what LiveTalk knows about a voice beyond what the
    /// TTS engine stores in <c>voice.json</c>. Written last, so its presence
    /// marks the folder complete.
    /// </summary>
    internal class VoiceMeta
    {
        public string id;
        [JsonConverter(typeof(StringEnumConverter))] public VoiceKind kind;
        public string sampleText;
        [JsonConverter(typeof(StringEnumConverter))] public Gender gender;
        [JsonConverter(typeof(StringEnumConverter))] public Pitch pitch;
        [JsonConverter(typeof(StringEnumConverter))] public Speed speed;
        public string instruct;
        public DateTime createdUtc;
        public string version = "2.0";

        public const string FileName = "voice.meta.json";
    }

    /// <summary>
    /// A speaker. Either designed from a description or cloned from a
    /// recording, saved once, and reloaded thereafter.
    ///
    /// <para><b>Identity.</b> <see cref="Id"/> depends on <see cref="Kind"/>:
    /// a <see cref="VoiceKind.Designed"/> voice gets a fresh GUID per
    /// <see cref="LiveTalkAPI.DesignVoiceAsync"/> call, because VoiceDesign
    /// samples a new speaker every time and two rolls are two voices; a
    /// <see cref="VoiceKind.Cloned"/> voice is
    /// <see cref="HashUtils.GenerateClonedVoiceId"/> — a content hash of the
    /// reference PCM and its transcript — so locking the same take twice
    /// yields the same voice and <see cref="LiveTalkAPI.CloneVoiceAsync"/> is
    /// get-or-create.</para>
    ///
    /// <para><b>Storage.</b> <c>&lt;saveLocation&gt;/voices/&lt;Id&gt;/</c>,
    /// in the TTS engine's own layout (<c>voice.json</c>,
    /// <c>clone_prompt.bin</c>, <c>reference.wav</c>, <c>sample.wav</c>) plus
    /// <c>voice.meta.json</c> with the kind, the sample text and the design
    /// parameters. <c>voice.meta.json</c> is written last; a folder without
    /// it is incomplete.</para>
    ///
    /// <para><b>Sample.</b> <see cref="Sample"/> is the take that represents
    /// this voice: for a designed voice, <see cref="SampleText"/> rendered at
    /// creation (a host can later lock it by cloning it); for a clone, the
    /// reference recording itself, with <see cref="SampleText"/> its
    /// transcript. Null only for a voice loaded from a folder that has no
    /// <c>sample.wav</c>.</para>
    ///
    /// <para>Speech audio is cached on <see cref="Id"/> + text
    /// (<see cref="HashUtils.GenerateSpeechCacheKey"/>), not on the character
    /// that speaks it.</para>
    /// </summary>
    public sealed class Voice
    {
        /// <summary>GUID for a designed voice; content hash of reference + transcript for a clone. See the class remarks.</summary>
        public string Id { get; }

        /// <summary>Designed or cloned.</summary>
        public VoiceKind Kind { get; }

        /// <summary>
        /// The rendered take (designed) or the reference recording (cloned).
        /// Null only for a loaded voice whose folder has no <c>sample.wav</c>.
        /// </summary>
        public AudioClip Sample { get; internal set; }

        /// <summary>What <see cref="Sample"/> says: the sample text or the clone transcript.</summary>
        public string SampleText { get; }

        /// <summary>Design parameter. Meaningful for <see cref="VoiceKind.Designed"/>; default for a clone.</summary>
        public Gender Gender { get; }

        /// <summary>Design parameter. Meaningful for <see cref="VoiceKind.Designed"/>; default for a clone.</summary>
        public Pitch Pitch { get; }

        /// <summary>Design parameter. Meaningful for <see cref="VoiceKind.Designed"/>; default for a clone.</summary>
        public Speed Speed { get; }

        /// <summary>Free-text VoiceDesign notes the voice was designed with, or null.</summary>
        public string Instruct { get; }

        /// <summary>
        /// True when this voice lives inline in a pre-2.0 character folder
        /// (<c>&lt;character&gt;/voice/</c>) rather than under <c>voices/</c>.
        /// Its <see cref="Id"/> is then the legacy character's id, and it
        /// cannot be shared or deleted on its own.
        /// </summary>
        public bool IsLegacy { get; }

        internal string Folder { get; }
        internal QwenVoice Loaded { get; }

        private Voice(string id, VoiceKind kind, string folder, QwenVoice loaded, VoiceMeta meta, bool isLegacy)
        {
            Id = id;
            Kind = kind;
            Folder = folder;
            Loaded = loaded;
            IsLegacy = isLegacy;
            SampleText = meta?.sampleText;
            Gender = meta?.gender ?? default;
            Pitch = meta?.pitch ?? default;
            Speed = meta?.speed ?? default;
            Instruct = meta?.instruct;
        }

        #region Design

        /// <summary>
        /// Designs a new speaker and renders <paramref name="sampleText"/> as its
        /// sample. Throws on every failure; the staging folder is removed in
        /// <c>finally</c> so nothing half-written survives.
        /// </summary>
        internal static async Task<Voice> DesignAsync(
            LiveTalkAPI api, Gender gender, Pitch pitch, Speed speed, string instruct, string sampleText)
        {
            if (api == null)
                throw new InvalidOperationException("LiveTalkAPI not initialized. Call LiveTalkAPI.Initialize() first.");
            if (string.IsNullOrWhiteSpace(sampleText))
                throw new ArgumentException(
                    "A designed voice needs sample text to render: that rendering is the voice's Sample, " +
                    "which is what a host auditions and later locks as a clone reference.", nameof(sampleText));

            string id = Guid.NewGuid().ToString("N");
            string finalFolder = LiveTalkStorage.VoiceFolder(id);
            string staging = LiveTalkStorage.BeginStaging(finalFolder);
            bool committed = false;

            Logger.Log($"[Voice] Designing voice {id}: {gender}/{pitch}/{speed}");
            try
            {
                QwenVoice designed;
                SpeechResult sample;

                // The engine is single-tenant; a character speaking at the
                // same time waits its turn rather than racing the talker.
                await api.VoiceQueue.AcquireAsync();
                try
                {
                    designed = await QwenTts.CreateDesignedVoiceAsync(
                        new VoiceDesignSpec(VoiceInstruct.Compose(gender, pitch, speed, instruct)))
                        ?? throw new InvalidOperationException(
                            $"Failed to design a voice ({gender}/{pitch}/{speed}) — the TTS engine returned no voice. " +
                            "Check that the VoiceDesign model is present and initialized.");

                    // A designed voice has no inherent audio; this rendering is
                    // its Sample, at the engine's native rate so it is a usable
                    // clone reference (the speaker encoder reads mel to 12 kHz).
                    sample = await designed.SpeakAsync(sampleText);
                }
                finally
                {
                    api.VoiceQueue.Release();
                }

                await designed.SaveAsync(staging, sample);
                var meta = new VoiceMeta
                {
                    id = id,
                    kind = VoiceKind.Designed,
                    sampleText = sampleText,
                    gender = gender,
                    pitch = pitch,
                    speed = speed,
                    instruct = instruct,
                    createdUtc = DateTime.UtcNow,
                };
                await WriteMetaAsync(staging, meta);

                committed = LiveTalkStorage.CommitStaging(staging, finalFolder);
                if (!committed)
                    throw new InvalidOperationException($"Voice folder {finalFolder} appeared while designing; GUID collision?");

                var voice = new Voice(id, VoiceKind.Designed, finalFolder, designed, meta, isLegacy: false)
                {
                    Sample = sample.ToAudioClip($"voice_{id}_sample"),
                };
                Logger.Log($"[Voice] Designed voice {id} saved to {finalFolder}");
                return voice;
            }
            finally
            {
                if (!committed)
                    LiveTalkStorage.DeleteFolder(staging);
            }
        }

        #endregion

        #region Clone

        /// <summary>
        /// Clones <paramref name="reference"/>, or loads the existing clone of
        /// exactly that reference + transcript. Throws on every failure; the
        /// staging folder is removed in <c>finally</c>.
        /// </summary>
        internal static async Task<Voice> CloneAsync(LiveTalkAPI api, AudioClip reference, string transcript)
        {
            if (api == null)
                throw new InvalidOperationException("LiveTalkAPI not initialized. Call LiveTalkAPI.Initialize() first.");
            if (reference == null)
                throw new ArgumentNullException(nameof(reference), "A cloned voice needs a reference recording.");

            // Main-thread: AudioClip.GetData. Also what the id hashes.
            var referencePcm = ClipToSpeechResult(reference);
            string id = HashUtils.GenerateClonedVoiceId(referencePcm.Pcm, referencePcm.SampleRate, transcript)
                        ?? throw new ArgumentException("Reference recording is empty.", nameof(reference));
            string finalFolder = LiveTalkStorage.VoiceFolder(id);

            if (Directory.Exists(finalFolder))
            {
                if (IsComplete(finalFolder))
                {
                    Logger.Log($"[Voice] Reusing cloned voice {id} for this reference — skipping the encoders.");
                    var existing = await LoadAsync(finalFolder, id, isLegacy: false, fallbackMeta: null);
                    // The caller's clip is the same audio as sample.wav; prefer
                    // it so nothing depends on the wav round trip.
                    existing.Sample = reference;
                    return existing;
                }
                Logger.LogWarning($"[Voice] Voice folder {finalFolder} is incomplete; rebuilding.");
                LiveTalkStorage.DeleteFolder(finalFolder);
            }

            string staging = LiveTalkStorage.BeginStaging(finalFolder);
            bool committed = false;

            Logger.Log($"[Voice] Cloning voice {id} from '{reference.name}' ({reference.length:F1}s)" +
                       (string.IsNullOrWhiteSpace(transcript) ? " without transcript (x-vector only)" : ""));
            try
            {
                QwenVoice cloned;
                await api.VoiceQueue.AcquireAsync();
                try
                {
                    // Async because a cold clone loads the Base tables plus two
                    // reference encoders - tens of seconds of main-thread stall
                    // otherwise.
                    cloned = await QwenTts.CreateClonedVoiceAsync(reference, transcript)
                        ?? throw new InvalidOperationException(
                            $"Failed to clone the voice from '{reference.name}' — the TTS engine returned no voice.");
                }
                finally
                {
                    api.VoiceQueue.Release();
                }

                // The reference recording *is* this voice's sample, so it is
                // saved as one. SaveAsync also stores the derived clone prompt,
                // which is what stops the next load re-running both encoders.
                await cloned.SaveAsync(staging, referencePcm);
                var meta = new VoiceMeta
                {
                    id = id,
                    kind = VoiceKind.Cloned,
                    sampleText = transcript,
                    createdUtc = DateTime.UtcNow,
                };
                await WriteMetaAsync(staging, meta);

                committed = LiveTalkStorage.CommitStaging(staging, finalFolder);
                if (!committed)
                {
                    // Another clone of the same take finished first. Its folder
                    // is byte-for-byte what ours would have been.
                    var winner = await LoadAsync(finalFolder, id, isLegacy: false, fallbackMeta: null);
                    winner.Sample = reference;
                    return winner;
                }

                var voice = new Voice(id, VoiceKind.Cloned, finalFolder, cloned, meta, isLegacy: false)
                {
                    Sample = reference,
                };
                Logger.Log($"[Voice] Cloned voice {id} saved to {finalFolder}");
                return voice;
            }
            finally
            {
                if (!committed)
                    LiveTalkStorage.DeleteFolder(staging);
            }
        }

        /// <summary>PCM view of a clip (mono mixdown), so it can be saved as a voice sample and hashed.</summary>
        internal static SpeechResult ClipToSpeechResult(AudioClip clip)
        {
            var interleaved = new float[clip.samples * clip.channels];
            clip.GetData(interleaved, 0);
            if (clip.channels <= 1)
                return new SpeechResult(interleaved, clip.frequency);

            var mono = new float[clip.samples];
            for (int i = 0; i < clip.samples; i++)
            {
                float sum = 0f;
                for (int c = 0; c < clip.channels; c++)
                    sum += interleaved[i * clip.channels + c];
                mono[i] = sum / clip.channels;
            }
            return new SpeechResult(mono, clip.frequency);
        }

        #endregion

        #region Load

        /// <summary>A voice folder is complete when the engine manifest and LiveTalk's meta both exist.</summary>
        internal static bool IsComplete(string folder) =>
            File.Exists(Path.Combine(folder, "voice.json")) && File.Exists(Path.Combine(folder, VoiceMeta.FileName));

        /// <summary>
        /// Loads a voice folder — <c>voices/&lt;id&gt;</c>, or a pre-2.0
        /// character's inline <c>voice/</c>. Throws on a missing folder, a
        /// missing <c>voice.json</c>, or a faulted engine load: a voice that
        /// cannot speak is not loaded. An unreadable <c>sample.wav</c> is
        /// logged and <see cref="Sample"/> stays null.
        /// </summary>
        /// <param name="fallbackMeta">
        /// Used when the folder has no <c>voice.meta.json</c> (legacy inline
        /// voice). Null infers kind and text from the engine manifest.
        /// </param>
        internal static async Task<Voice> LoadAsync(string folder, string id, bool isLegacy, VoiceMeta fallbackMeta)
        {
            if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            {
                throw new DirectoryNotFoundException(
                    $"Voice folder not found: {folder}. The voice was never saved, or its creation failed part-way.");
            }

            string voiceJson = Path.Combine(folder, "voice.json");
            if (!File.Exists(voiceJson))
            {
                throw new FileNotFoundException(
                    $"Voice folder has no voice.json: {voiceJson}. Voice design or clone did not complete; recreate the voice.",
                    voiceJson);
            }

            // Restores the stored clone prompt when there is one, so this does
            // not re-run the speaker and tokenizer encoders. A fault (engine
            // not initialized, corrupt prompt, missing reference) throws.
            var loaded = await QwenTts.LoadVoiceAsync(folder)
                ?? throw new InvalidOperationException($"The TTS engine returned no voice for {folder}.");

            VoiceMeta meta = null;
            string metaPath = Path.Combine(folder, VoiceMeta.FileName);
            if (File.Exists(metaPath))
            {
                try
                {
                    meta = JsonConvert.DeserializeObject<VoiceMeta>(await File.ReadAllTextAsync(metaPath));
                }
                catch (Exception ex)
                {
                    Logger.LogWarning($"[Voice] Unreadable {metaPath}, inferring from the engine manifest: {ex.Message}");
                }
            }
            meta ??= fallbackMeta ?? new VoiceMeta
            {
                id = id,
                kind = loaded.IsCloned ? VoiceKind.Cloned : VoiceKind.Designed,
                sampleText = loaded.IsCloned ? loaded.ReferenceText : null,
                instruct = loaded.Instruct,
            };
            // The engine manifest is authoritative about what the voice is; a
            // fallback (legacy) meta only knows what the character was asked for.
            if (loaded.IsCloned && meta.kind != VoiceKind.Cloned)
            {
                meta.kind = VoiceKind.Cloned;
                meta.sampleText = string.IsNullOrEmpty(loaded.ReferenceText) ? meta.sampleText : loaded.ReferenceText;
            }
            var kind = meta.kind;

            var voice = new Voice(id, kind, folder, loaded, meta, isLegacy);

            // The rendered sample is optional; a designed voice only has one
            // if a take was saved with it.
            string samplePath = Path.Combine(folder, "sample.wav");
            if (File.Exists(samplePath))
            {
                try
                {
                    voice.Sample = await AudioFileIO.LoadClipAsync(samplePath);
                    if (voice.Sample != null)
                        voice.Sample.name = $"voice_{id}_sample";
                }
                catch (Exception ex)
                {
                    Logger.LogWarning($"[Voice] Voice sample unreadable, continuing without it: {samplePath}: {ex.Message}");
                }
            }

            Logger.LogVerbose($"[Voice] Loaded {kind} voice {id} from {folder}");
            return voice;
        }

        private static Task WriteMetaAsync(string folder, VoiceMeta meta) =>
            File.WriteAllTextAsync(Path.Combine(folder, VoiceMeta.FileName),
                JsonConvert.SerializeObject(meta, Formatting.Indented));

        #endregion
    }
}
