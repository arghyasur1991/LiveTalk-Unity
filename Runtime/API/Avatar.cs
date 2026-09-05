using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Video;
using Newtonsoft.Json;

namespace LiveTalk.API
{
    using Core;
    using Utils;

    internal class ProcessFramesResult
    {
        public List<Texture2D> GeneratedFrames { get; set; } = new List<Texture2D>();
        public List<string> GeneratedFramePaths { get; set; } = new List<string>();
    }

    /// <summary>
    /// Data structures for face data JSON deserialization
    /// </summary>
    [Serializable]
    internal class FaceDataContainer
    {
        public FaceRegionData[] faceRegions;
    }

    [Serializable]
    internal class FaceRegionData
    {
        public bool hasFace;
        public BoundingBoxData boundingBox;
        public BoundingBoxData adjustedFaceBbox;
        public BoundingBoxData cropBox;
        public TextureFilesData textureFiles;
        public TextureDimensionsData textureDimensions;
    }

    [Serializable]
    internal class BoundingBoxData
    {
        public float x;
        public float y;
        public float width;
        public float height;
        public float z; // For Vector4 data
        public float w; // For Vector4 data
    }

    [Serializable]
    internal class TextureFilesData
    {
        public string croppedFace;
        public string faceLarge;
        public string segmentationMask;
        public string original;
        public string maskSmall;
        public string fullMask;
        public string boundaryMask;
        public string blurredMask;
    }

    [Serializable]
    internal class TextureDimensionsData
    {
        public TextureDimension croppedFace;
        public TextureDimension faceLarge;
        public TextureDimension segmentationMask;
        public TextureDimension original;
        public TextureDimension maskSmall;
        public TextureDimension fullMask;
        public TextureDimension boundaryMask;
        public TextureDimension blurredMask;
    }

    [Serializable]
    internal class TextureDimension
    {
        public int width;
        public int height;
    }

    /// <summary>
    /// Data for a specific expression including frames, latents, and face data
    /// </summary>
    internal class ExpressionData
    {
        public AvatarData Data { get; set; } = new AvatarData();
        public string ExpressionName { get; set; }
    }

    /// <summary>
    /// <c>avatar.json</c>: written last, so its presence means every
    /// expression finished. Records the mode so a load does not have to infer
    /// it from the folder count.
    /// </summary>
    internal class AvatarManifest
    {
        public string id;
        [JsonConverter(typeof(Newtonsoft.Json.Converters.StringEnumConverter))]
        public CreationMode mode;
        public string[] expressions;
        public DateTime createdUtc;
        public string version = "2.0";

        /// <summary>Frame rate the driving frames were rendered at. Absent in folders built before motion editing (played at 25).</summary>
        public float? fps;

        /// <summary>True when each expression's frames run seamlessly from last back to first. Absent / false: ping-pong.</summary>
        public bool? loopable;

        /// <summary>
        /// Version of the driving-motion pipeline that built the frames. Part
        /// of the avatar id, so a pipeline change rebuilds every avatar rather
        /// than half-matching an old folder. See <see cref="Avatar.MotionPipelineVersion"/>.
        /// </summary>
        public int? motionPipelineVersion;

        /// <summary>
        /// Fingerprint of the bundled driving clips the frames were rendered
        /// from (<see cref="Avatar.DrivingClipsHash"/>). Part of the avatar
        /// id, so replacing a clip rebuilds every avatar instead of serving
        /// frames driven by the old footage. Absent in folders built before it.
        /// </summary>
        public string drivingClipsHash;

        public const string FileName = "avatar.json";
    }

    /// <summary>
    /// A face that can be animated: the source portrait plus the driving
    /// frames, latents and face crops that LivePortrait and MuseTalk need to
    /// lip-sync it. Building one takes minutes and hundreds of MB; loading one
    /// takes seconds. Immutable once built.
    ///
    /// <para><b>Identity.</b> <see cref="Id"/> is a content hash of the source
    /// image bytes and the expression set (<see cref="Mode"/>), from
    /// <see cref="HashUtils.GenerateAvatarId"/>. The same portrait built the
    /// same way always has the same id, so
    /// <see cref="LiveTalkAPI.CreateAvatarAsync"/> is get-or-create and the
    /// avatar folder is its own cache — no copying, no pruning.</para>
    ///
    /// <para><b>Storage.</b>
    /// <c>&lt;saveLocation&gt;/avatars/&lt;Id&gt;/</c> containing
    /// <c>image.png</c>, <c>avatar.json</c> and
    /// <c>drivingFrames/expression-N/</c> (numbered PNG frames,
    /// <c>latents.bin</c>, <c>faces.json</c>, <c>textures/…</c>) for each
    /// expression in the set. <c>avatar.json</c> is written last; a folder
    /// without it is incomplete and is rebuilt on the next create.</para>
    ///
    /// <para>Characters reference an avatar by id (<see cref="Character.Avatar"/>)
    /// and never copy it. Several characters — or several voices for the same
    /// face — share one avatar folder.</para>
    /// </summary>
    public sealed class Avatar
    {
        /// <summary>
        /// Content hash of the source image and expression set. See the class
        /// remarks for the rule. For an avatar loaded from a pre-2.0 inline
        /// character folder this is that character's id.
        /// </summary>
        public string Id { get; }

        /// <summary>The source portrait. Loaded from <c>image.png</c>; null only if that file is missing.</summary>
        public Texture2D Image { get; internal set; }

        /// <summary>
        /// Which expressions were generated. <see cref="CreationMode.VoiceOnly"/>
        /// is an image with no driving frames — usable as a thumbnail, not
        /// animatable.
        /// </summary>
        public CreationMode Mode { get; }

        /// <summary>Expression indices that can be passed to <see cref="Character.SpeakAsync"/>.</summary>
        public IReadOnlyCollection<int> ExpressionIndices => LoadedExpressions.Keys;

        /// <summary>True when at least one expression's frames and latents are loaded.</summary>
        public bool CanAnimate => LoadedExpressions.Count > 0;

        /// <summary>
        /// True when this avatar lives inline in a pre-2.0 character folder
        /// rather than under <c>avatars/</c>. Such an avatar cannot be shared
        /// or deleted on its own.
        /// </summary>
        public bool IsLegacy { get; }

        /// <summary>
        /// Frame rate the driving frames play at. Frames built by the motion
        /// pipeline (<see cref="MotionPipelineVersion"/> 2) are retimed to
        /// <see cref="DrivingMotionOptions.TargetFps"/>, 25 by default — the
        /// rate lip-sync frames are generated at, so idle and speech share one
        /// clock. Folders built before that record no rate and were always
        /// displayed at 25, so 25 is reported for them too.
        /// </summary>
        public float FrameRate { get; private set; } = DefaultFrameRate;

        /// <summary>
        /// True when every expression's frames run seamlessly from the last
        /// back to the first (see <c>MotionSequence.MakeLoopable</c>), so a
        /// player should loop them forward rather than ping-pong. False for
        /// folders built before the motion pipeline, which have to be
        /// ping-ponged to avoid a hard cut at the wrap.
        /// </summary>
        public bool IsLoopable { get; private set; }

        internal string Folder { get; }
        internal Dictionary<int, ExpressionData> LoadedExpressions { get; } = new Dictionary<int, ExpressionData>();

        internal string DrivingFramesFolder => Path.Combine(Folder, DrivingFramesFolderName);
        internal string ExpressionFolder(int index) => Path.Combine(DrivingFramesFolder, $"expression-{index}");

        internal const string ImageFileName = "image.png";
        internal const string DrivingFramesFolderName = "drivingFrames";
        internal const float DefaultFrameRate = 25f;

        /// <summary>
        /// Version of the driving-motion pipeline. Hashed into the avatar id
        /// (<see cref="Signature"/>), so bumping it rebuilds every avatar into
        /// a new folder instead of reusing frames built the old way.
        /// <list type="bullet">
        /// <item>1 (implicit, pre-2.1): one frame per driving frame at the clip's native rate, not loopable.</item>
        /// <item>2: motion resampled to 25 fps and crossfaded into a seamless forward loop before rendering.</item>
        /// <item>3: driving frames are face-cropped (fixed from frame 0) before motion extraction and relative scale is bounded, so the head no longer jumps in size and expressions transfer at full strength.</item>
        /// <item>4: ScaleTransfer=0 (head size pinned), gain 2.0.</item>
        /// <item>5: the face crop is upright. <c>FaceAnalysis.ParsePt2FromPtX</c> had applied the
        /// 106-point eye/lip indices to the 203-point tracker output, rotating every source and
        /// driving crop ~40° before the motion extractor saw it (expressions read weak, eyes
        /// overshot on blinks, motion applied along a rotated axis). Gain re-tuned to 1.4 against
        /// upright crops.</item>
        /// </list>
        /// </summary>
        internal const int MotionPipelineVersion = 5;

        internal static readonly string[] AllExpressionNames =
            { "talk-neutral", "approve", "disapprove", "smile", "sad", "surprised", "confused" };

        /// <summary>
        /// The motion edit every avatar is built with. Deliberately not a
        /// knob: the numbers are part of the folder format, so changing them
        /// means bumping <see cref="MotionPipelineVersion"/>. A new instance
        /// per call because the API fills <see cref="DrivingMotionOptions.SourceFps"/>
        /// in from each expression's clip.
        /// </summary>
        private static DrivingMotionOptions DrivingMotion(string expression) =>
            new()
            {
                TargetFps = DefaultFrameRate,
                LoopBlendSeconds = 0.4f,
                ExpressionGain = ExpressionGainFor(expression),
            };

        /// <summary>
        /// Per-expression <see cref="DrivingMotionOptions.ExpressionGain"/>.
        /// Chosen on the reference (upright-crop) pipeline rendering the bundled
        /// clips onto a neutral portrait: 1.0 already reads, 1.4 is clearly
        /// expressive with natural blinks, 1.8 is strong but starts to look
        /// pushed on the surprised mouth. The earlier 2.0 was tuned against the
        /// rotated crop (pipeline version 4) and over-drives an upright one.
        /// Part of <see cref="Signature"/>: changing a value rebuilds.
        /// </summary>
        internal static float ExpressionGainFor(string expression) => 1.4f;

        private static string ExpressionGainSignature(CreationMode mode) =>
            string.Join(",", ExpressionsFor(mode).Select(e => e + "=" + ExpressionGainFor(e).ToString("0.##", System.Globalization.CultureInfo.InvariantCulture)));

        private Avatar(string id, CreationMode mode, string folder, Texture2D image, bool isLegacy)
        {
            Id = id;
            Mode = mode;
            Folder = folder;
            Image = image;
            IsLegacy = isLegacy;
        }

        /// <summary>Expression names generated for a mode, in index order.</summary>
        internal static string[] ExpressionsFor(CreationMode mode) => mode switch
        {
            CreationMode.VoiceOnly => Array.Empty<string>(),
            CreationMode.SingleExpression => new[] { AllExpressionNames[0] },
            _ => AllExpressionNames,
        };

        /// <summary>
        /// The expression-set half of the avatar id. Carries the motion
        /// pipeline version and the driving-clip fingerprint, so a pipeline
        /// change or a new set of clips gives every avatar a new id: an old
        /// folder is neither reused nor half-matched, and the two generations
        /// can coexist on disk until the old one is deleted.
        /// </summary>
        internal static string Signature(CreationMode mode) =>
            mode + ":" + string.Join(",", ExpressionsFor(mode)) + ";motion=v" + MotionPipelineVersion
            + ";gain=" + ExpressionGainSignature(mode)
            + ";clips=" + DrivingClipsHash;

        private static string _drivingClipsHash;

        /// <summary>
        /// Fingerprint of the bundled driving clips, computed once per session
        /// from every <c>Resources/driving/*</c> <see cref="VideoClip"/>. Hashes
        /// what a <see cref="VideoClip"/> exposes in a player as well as in the
        /// editor — name, frame count, frame rate, size, length — which is what
        /// changes when a clip is re-authored (the raw bytes are not readable
        /// from an imported VideoClip at runtime). All seven clips are covered
        /// whatever the mode, so a change to any of them rebuilds every avatar.
        /// </summary>
        internal static string DrivingClipsHash
        {
            get
            {
                if (_drivingClipsHash == null)
                    _drivingClipsHash = ComputeDrivingClipsHash();
                return _drivingClipsHash;
            }
        }

        private static string ComputeDrivingClipsHash()
        {
            var sb = new System.Text.StringBuilder();
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            foreach (string expression in AllExpressionNames)
            {
                VideoClip clip = LoadDrivingVideoForExpression(expression);
                sb.Append(expression).Append('=');
                if (clip == null)
                {
                    sb.Append("missing;");
                    continue;
                }
                sb.Append(clip.frameCount).Append('@').Append(clip.frameRate.ToString("R", inv))
                  .Append(',').Append(clip.width).Append('x').Append(clip.height)
                  .Append(',').Append(clip.length.ToString("R", inv)).Append(';');
            }
            string hash = HashUtils.GenerateTextHash(sb.ToString()).Substring(0, 12);
            Logger.LogVerbose($"[Avatar] Driving clips fingerprint {hash} ({sb})");
            return hash;
        }

        /// <summary>Human name of an expression index, for logs.</summary>
        internal static string GetExpressionName(int index) =>
            index >= 0 && index < AllExpressionNames.Length ? AllExpressionNames[index] : $"expression-{index}";

        /// <summary>
        /// Encodes the image the way the avatar folder stores it, so the id
        /// hashes exactly the bytes that end up in <c>image.png</c>.
        /// </summary>
        internal static byte[] EncodeImage(Texture2D image)
        {
            var uncompressed = TextureUtils.ConvertToUncompressedTexture(image);
            try
            {
                return uncompressed.EncodeToPNG();
            }
            finally
            {
                if (uncompressed != image)
                    UnityEngine.Object.DestroyImmediate(uncompressed);
            }
        }

        /// <summary>
        /// A folder is complete when its manifest exists and every expression
        /// the manifest lists has frames, latents and face data.
        /// </summary>
        internal static bool IsComplete(string folder, out string reason)
        {
            reason = null;
            string manifestPath = Path.Combine(folder, AvatarManifest.FileName);
            if (!File.Exists(manifestPath))
            {
                reason = $"no {AvatarManifest.FileName}";
                return false;
            }

            AvatarManifest manifest;
            try
            {
                manifest = JsonConvert.DeserializeObject<AvatarManifest>(File.ReadAllText(manifestPath));
            }
            catch (Exception ex)
            {
                reason = $"unreadable {AvatarManifest.FileName}: {ex.Message}";
                return false;
            }
            if (manifest?.expressions == null)
            {
                reason = $"{AvatarManifest.FileName} lists no expressions";
                return false;
            }

            for (int i = 0; i < manifest.expressions.Length; i++)
            {
                string expressionFolder = Path.Combine(folder, DrivingFramesFolderName, $"expression-{i}");
                if (!Directory.Exists(expressionFolder))
                {
                    reason = $"missing {expressionFolder}";
                    return false;
                }
                if (!File.Exists(Path.Combine(expressionFolder, "latents.bin"))
                    || !File.Exists(Path.Combine(expressionFolder, "faces.json")))
                {
                    reason = $"expression-{i} has no latents.bin / faces.json";
                    return false;
                }
                if (!Directory.EnumerateFiles(expressionFolder, "*.png").Any())
                {
                    reason = $"expression-{i} has no frames";
                    return false;
                }
            }
            return true;
        }

        #region Create

        /// <summary>
        /// Unguarded get-or-create. Faults propagate out of the iterator so the
        /// caller's <see cref="TaskYield.Guard"/> routes them to its onError and
        /// never reaches onComplete. A run that fails part-way deletes its
        /// staging folder in <c>finally</c>; the final folder only ever holds a
        /// finished avatar.
        /// </summary>
        internal static IEnumerator CreateOrLoadCore(Texture2D image, CreationMode mode, Action<Avatar> onComplete)
        {
            var api = LiveTalkAPI.Instance ?? throw new InvalidOperationException(
                "LiveTalkAPI not initialized. Call LiveTalkAPI.Initialize() first.");
            if (image == null)
                throw new ArgumentNullException(nameof(image), "An avatar needs a source image.");

            var start = System.Diagnostics.Stopwatch.StartNew();
            byte[] imageBytes = EncodeImage(image);
            string[] expressions = ExpressionsFor(mode);
            string id = HashUtils.GenerateAvatarId(imageBytes, Signature(mode))
                        ?? throw new InvalidOperationException("Could not encode the source image.");
            string finalFolder = LiveTalkStorage.AvatarFolder(id);

            if (Directory.Exists(finalFolder))
            {
                if (IsComplete(finalFolder, out string reason))
                {
                    Logger.Log($"[Avatar] Reusing avatar {id} ({mode}) — skipping avatar preprocessing.");
                    yield return LoadCore(finalFolder, id, mode, isLegacy: false, onComplete);
                    yield break;
                }
                Logger.LogWarning($"[Avatar] Avatar folder {finalFolder} is incomplete ({reason}); rebuilding.");
                LiveTalkStorage.DeleteFolder(finalFolder);
            }

            Logger.Log($"[Avatar] Creating avatar {id} ({mode}, {expressions.Length} expression(s)) in {finalFolder}");
            string staging = LiveTalkStorage.BeginStaging(finalFolder);
            bool committed = false;
            try
            {
                string imagePath = Path.Combine(staging, ImageFileName);
                yield return TaskYield.Wait(File.WriteAllBytesAsync(imagePath, imageBytes),
                    $"Avatar.Create write {imagePath}");

                string drivingFramesFolder = Path.Combine(staging, DrivingFramesFolderName);
                Directory.CreateDirectory(drivingFramesFolder);

                for (int expressionIndex = 0; expressionIndex < expressions.Length; expressionIndex++)
                {
                    string expression = expressions[expressionIndex];
                    string expressionFolder = Path.Combine(drivingFramesFolder, $"expression-{expressionIndex}");
                    Directory.CreateDirectory(expressionFolder);

                    Logger.Log($"[Avatar] Processing expression: {expression} (index: {expressionIndex})");

                    // A missing driving video is a failed avatar, not a shorter
                    // one: the folder is contractually complete or absent.
                    VideoClip drivingVideo = LoadDrivingVideoForExpression(expression)
                        ?? throw new FileNotFoundException(
                            $"No driving video for expression '{expression}'. Expected a VideoClip at " +
                            $"Resources/driving/{expression} (or LiveTalk/driving/{expression}).");

                    yield return ProcessExpressionCoroutine(image, expression, drivingVideo, expressionFolder, api);
                }

                // Manifest last: its presence is what marks the folder complete.
                var motion = DrivingMotion("talk-neutral");   // fps / loopable are the same for every expression; only gain differs
                var manifest = new AvatarManifest
                {
                    id = id,
                    mode = mode,
                    expressions = expressions,
                    createdUtc = DateTime.UtcNow,
                    fps = motion.TargetFps > 0f ? motion.TargetFps : DefaultFrameRate,
                    loopable = motion.Loopable,
                    motionPipelineVersion = MotionPipelineVersion,
                    drivingClipsHash = DrivingClipsHash,
                };
                string manifestPath = Path.Combine(staging, AvatarManifest.FileName);
                yield return TaskYield.Wait(
                    File.WriteAllTextAsync(manifestPath, JsonConvert.SerializeObject(manifest, Formatting.Indented)),
                    $"Avatar.Create write {manifestPath}");

                committed = LiveTalkStorage.CommitStaging(staging, finalFolder);
            }
            finally
            {
                // Faulted, or the host stopped the coroutine: nothing half-built
                // may remain. (CommitStaging removed the staging folder itself
                // when another create of the same avatar won the race.)
                if (!committed)
                    LiveTalkStorage.DeleteFolder(staging);
            }

            Logger.Log($"[Avatar] Avatar {id} created in {start.Elapsed.TotalSeconds:F1}s");
            yield return LoadCore(finalFolder, id, mode, isLegacy: false, onComplete);
        }

        /// <summary>
        /// Process a single expression with coroutines to handle frame streaming
        /// </summary>
        private static IEnumerator ProcessExpressionCoroutine(
            Texture2D image,
            string expression,
            VideoClip drivingVideo,
            string expressionFolder,
            LiveTalkAPI liveTalkAPI)
        {
            var videoPlayer = liveTalkAPI.Object.GetComponent<VideoPlayer>();
            videoPlayer.clip = drivingVideo;
            videoPlayer.isLooping = false;
            videoPlayer.playOnAwake = false;
            videoPlayer.skipOnDrop = false;
            videoPlayer.Prepare();
            yield return new WaitUntil(() => videoPlayer.isPrepared);

            // Generate animated textures using LivePortrait, with the driving
            // motion retimed to the canonical rate and made loopable first.
            var outputStream = liveTalkAPI.GenerateAnimatedTexturesAsync(image, videoPlayer, DrivingMotion(expression));

            // Process frames
            var processResult = new ProcessFramesResult();
            yield return ProcessFramesCoroutine(outputStream, expressionFolder, processResult, liveTalkAPI);
            videoPlayer.clip = null;

            // The LivePortrait producer marks its stream finished on a fault
            // too (so the loop above exits); a truncated expression is a
            // failed avatar, not a shorter one.
            if (outputStream.Error != null)
            {
                throw new InvalidOperationException(
                    $"Driving-frame generation failed for expression '{expression}': {outputStream.Error.Message}",
                    outputStream.Error);
            }

            Logger.LogVerbose($"[Avatar] Generated and saved {processResult.GeneratedFrames.Count + processResult.GeneratedFramePaths.Count} frames for expression: {expression}");

            // Generate and save cache data (latents + face data). A fault here
            // — the MuseTalk preprocess models failing to load, usually —
            // rethrows and fails the creation instead of leaving an
            // expression folder without latents.bin / faces.json.
            yield return TaskYield.Wait(GenerateAndSaveCacheData(expressionFolder, processResult, liveTalkAPI),
                $"Avatar.GenerateAndSaveCacheData {expression}");

            if (liveTalkAPI.Config.MemoryUsage == MemoryUsage.Optimal)
            {
                yield return new WaitForSeconds(2f); // Wait for GC to complete
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);
            }
        }

        /// <summary>
        /// Process frame stream using coroutines
        /// </summary>
        private static IEnumerator ProcessFramesCoroutine(
            FrameStream outputStream,
            string expressionFolder,
            ProcessFramesResult result,
            LiveTalkAPI liveTalkAPI)
        {
            int frameIndex = 0;

            // Process frames as they become available using coroutine pattern
            while (outputStream.HasMoreFrames)
            {
                var awaiter = outputStream.WaitForNext();
                yield return awaiter;

                if (awaiter.Texture != null)
                {
                    // Save LivePortrait generated frames as numbered PNGs (these are the driving frames)
                    string frameFileName = Path.Combine(expressionFolder, $"{frameIndex:D5}.png");
                    byte[] pngData = awaiter.Texture.EncodeToPNG();
                    yield return TaskYield.Wait(File.WriteAllBytesAsync(frameFileName, pngData),
                        $"Avatar.ProcessFrames write {frameFileName}");

                    // Keep reference for cache generation
                    if (liveTalkAPI.Config.MemoryUsage != MemoryUsage.Optimal)
                    {
                        result.GeneratedFrames.Add(awaiter.Texture);
                    }
                    else
                    {
                        result.GeneratedFramePaths.Add(frameFileName);
                        UnityEngine.Object.DestroyImmediate(awaiter.Texture);
                    }
                    frameIndex++;
                }
            }
        }

        /// <summary>
        /// Load driving video clip for the specified expression
        /// </summary>
        private static VideoClip LoadDrivingVideoForExpression(string expression)
        {
            // Try to load from Resources folder
            string[] possiblePaths =
            {
                $"driving/{expression}",
                $"LiveTalk/driving/{expression}",
                expression
            };

            foreach (string path in possiblePaths)
            {
                var videoClip = Resources.Load<VideoClip>(path);
                if (videoClip != null)
                {
                    Logger.LogVerbose($"[Avatar] Loaded driving video: {path}");
                    return videoClip;
                }
            }

            Logger.LogWarning($"[Avatar] Could not find driving video for expression: {expression}");
            return null;
        }

        /// <summary>
        /// Generate and save cache data (latents and face data) for the processed frames using MuseTalkInference.
        /// Throws on every failure; a folder without latents / faces is not an avatar.
        /// </summary>
        private static async Task GenerateAndSaveCacheData(
            string expressionFolder, ProcessFramesResult processResult, LiveTalkAPI liveTalkAPI)
        {
            Logger.LogVerbose("[Avatar] Generating latents and face data...");

            var avatarData = await ProcessAvatarImagesWithMuseTalk(liveTalkAPI, processResult);

            if (avatarData == null || avatarData.Latents.Count == 0)
            {
                throw new InvalidOperationException(
                    "MuseTalk produced no avatar data for the generated driving frames. No fallback available.");
            }

            await SaveLatentsToFile(expressionFolder, avatarData.Latents);
            await SaveFaceDataToFile(expressionFolder, avatarData.FaceRegions);

            Logger.LogVerbose($"[Avatar] Saved {avatarData.Latents.Count} latents, {avatarData.FaceRegions.Count} face regions");
        }

        /// <summary>
        /// Process avatar images using MuseTalkInference public API to extract real latents and face data
        /// This uses the actual MuseTalk face analysis and VAE encoder pipeline - NO FALLBACKS
        /// </summary>
        private static async Task<AvatarData> ProcessAvatarImagesWithMuseTalk(LiveTalkAPI liveTalkAPI, ProcessFramesResult processResult)
        {
            Logger.LogVerbose("[Avatar] Processing avatar textures using MuseTalk pipeline");

            AvatarData avatarData;
            if (liveTalkAPI.Config.MemoryUsage != MemoryUsage.Optimal)
            {
                avatarData = await liveTalkAPI.MuseTalk.ProcessAvatarImages(processResult.GeneratedFrames);
            }
            else
            {
                avatarData = await liveTalkAPI.MuseTalk.ProcessAvatarImages(processResult.GeneratedFramePaths);
            }

            if (avatarData?.FaceRegions?.Count == 0 || avatarData?.Latents?.Count == 0)
            {
                throw new InvalidOperationException($"MuseTalk processing failed to generate valid avatar data. FaceRegions: {avatarData?.FaceRegions?.Count ?? 0}, Latents: {avatarData?.Latents?.Count ?? 0}");
            }

            Logger.LogVerbose($"[Avatar] MuseTalk processing completed: {avatarData.Latents.Count} latents, {avatarData.FaceRegions.Count} face regions");
            return avatarData;
        }

        /// <summary>
        /// Save latents data to binary file
        /// </summary>
        private static async Task SaveLatentsToFile(string expressionFolder, List<float[]> latents)
        {
            var latentsFile = Path.Combine(expressionFolder, "latents.bin");

            // Calculate total size needed
            int totalFloats = latents.Sum(latent => latent.Length);
            var allLatents = new float[totalFloats];

            // Combine all latent arrays into one
            int offset = 0;
            foreach (var latent in latents)
            {
                Array.Copy(latent, 0, allLatents, offset, latent.Length);
                offset += latent.Length;
            }

            // Convert to bytes and save
            var latentsBytes = new byte[allLatents.Length * sizeof(float)];
            Buffer.BlockCopy(allLatents, 0, latentsBytes, 0, latentsBytes.Length);
            await File.WriteAllBytesAsync(latentsFile, latentsBytes);

            Logger.LogVerbose($"[Avatar] Saved {latents.Count} latent arrays ({totalFloats} total floats) to {latentsFile}");
        }

        /// <summary>
        /// Save face data to JSON file and save all precomputed textures
        /// </summary>
        private static async Task SaveFaceDataToFile(string expressionFolder, List<FaceData> faceRegions)
        {
            var facesFile = Path.Combine(expressionFolder, "faces.json");
            var texturesFolder = Path.Combine(expressionFolder, "textures");

            // Create texture subfolders
            var subfolders = new[]
            {
                "cropped", "faceLarge", "segmentationMask", "original",
                "maskSmall", "fullMask", "boundaryMask", "blurredMask"
            };

            foreach (var subfolder in subfolders)
            {
                Directory.CreateDirectory(Path.Combine(texturesFolder, subfolder));
            }

            Logger.LogVerbose($"[Avatar] Saving face data with precomputed textures for {faceRegions.Count} face regions");

            // Process each face region and save all textures
            var faceDataForJson = new List<object>();

            for (int faceIndex = 0; faceIndex < faceRegions.Count; faceIndex++)
            {
                var face = faceRegions[faceIndex];

                // Save all precomputed textures for this face
                var texturePaths = await SaveFaceTextures(texturesFolder, face, faceIndex);

                // Create face data entry with texture file references
                var faceDataEntry = new
                {
                    faceIndex = faceIndex,
                    hasFace = face.HasFace,
                    boundingBox = new
                    {
                        x = face.BoundingBox.x,
                        y = face.BoundingBox.y,
                        width = face.BoundingBox.width,
                        height = face.BoundingBox.height
                    },
                    landmarks = face.Landmarks?.Select(l => new { x = l.x, y = l.y }).ToArray(),
                    adjustedFaceBbox = new
                    {
                        x = face.AdjustedFaceBbox.x,
                        y = face.AdjustedFaceBbox.y,
                        z = face.AdjustedFaceBbox.z,
                        w = face.AdjustedFaceBbox.w
                    },
                    cropBox = new
                    {
                        x = face.CropBox.x,
                        y = face.CropBox.y,
                        z = face.CropBox.z,
                        w = face.CropBox.w
                    },
                    textureDimensions = new
                    {
                        croppedFace = new { width = face.CroppedFaceTexture.width, height = face.CroppedFaceTexture.height },
                        original = new { width = face.OriginalTexture.width, height = face.OriginalTexture.height },
                        faceLarge = new { width = face.FaceLarge.width, height = face.FaceLarge.height },
                        segmentationMask = new { width = face.SegmentationMask.width, height = face.SegmentationMask.height },
                        maskSmall = new { width = face.MaskSmall.width, height = face.MaskSmall.height },
                        fullMask = new { width = face.FullMask.width, height = face.FullMask.height },
                        boundaryMask = new { width = face.BoundaryMask.width, height = face.BoundaryMask.height },
                        blurredMask = new { width = face.BlurredMask.width, height = face.BlurredMask.height }
                    },
                    // Reference to saved texture files
                    textureFiles = texturePaths
                };

                faceDataForJson.Add(faceDataEntry);
            }

            var faceDataJson = new
            {
                faceRegions = faceDataForJson.ToArray(),
                frameCount = faceRegions.Count,
                timestamp = DateTime.UtcNow,
                version = "1.0-complete",
                description = "Complete face data with all precomputed textures saved as PNG files"
            };

            string json = JsonConvert.SerializeObject(faceDataJson, Formatting.Indented);
            await File.WriteAllTextAsync(facesFile, json);

            Logger.LogVerbose($"[Avatar] Saved complete face data with textures for {faceRegions.Count} face regions to {facesFile}");
        }

        /// <summary>
        /// Save all precomputed textures for a single face region
        /// </summary>
        private static async Task<Dictionary<string, string>> SaveFaceTextures(string texturesFolder, FaceData face, int faceIndex)
        {
            var texturePaths = new Dictionary<string, string>();

            // Define texture mappings: texture data -> folder name -> filename
            var textureMap = new List<(Frame frame, string folder, string key)>
            {
                (face.CroppedFaceTexture, "cropped", "croppedFace"),
                (face.FaceLarge, "faceLarge", "faceLarge"),
                (face.SegmentationMask, "segmentationMask", "segmentationMask"),
                (face.OriginalTexture, "original", "original"),
                (face.MaskSmall, "maskSmall", "maskSmall"),
                (face.FullMask, "fullMask", "fullMask"),
                (face.BoundaryMask, "boundaryMask", "boundaryMask"),
                (face.BlurredMask, "blurredMask", "blurredMask")
            };

            foreach (var (frame, folder, key) in textureMap)
            {
                if (frame.data != null && frame.data.Length > 0)
                {
                    string filename = $"face_{faceIndex:D3}.bytes";
                    string folderPath = Path.Combine(texturesFolder, folder);
                    string fullPath = Path.Combine(folderPath, filename);

                    // Save as bytes array
                    await File.WriteAllBytesAsync(fullPath, frame.data);

                    // Store relative path for JSON reference
                    string relativePath = Path.Combine("textures", folder, filename).Replace('\\', '/');
                    texturePaths[key] = relativePath;
                }
                else
                {
                    texturePaths[key] = null; // Mark as missing/empty
                    Logger.LogWarning($"[Avatar] {key} texture data is null or empty for face {faceIndex}");
                }
            }

            return texturePaths;
        }

        #endregion

        #region Load

        /// <summary>
        /// Unguarded load of an avatar folder — <c>avatars/&lt;id&gt;</c>, or a
        /// pre-2.0 character folder whose <c>image.png</c> and
        /// <c>drivingFrames/</c> sit inline. Throws on an unreadable expression;
        /// an avatar with half its latents is not an avatar.
        /// </summary>
        /// <param name="modeHint">
        /// Used when the folder has no <c>avatar.json</c> (legacy). Null infers
        /// the mode from the number of expression folders.
        /// </param>
        internal static IEnumerator LoadCore(
            string folder, string id, CreationMode? modeHint, bool isLegacy, Action<Avatar> onComplete)
        {
            if (!Directory.Exists(folder))
                throw new DirectoryNotFoundException($"Avatar folder not found: {folder}");

            var start = System.Diagnostics.Stopwatch.StartNew();

            // Image
            Texture2D image = null;
            string imagePath = Path.Combine(folder, ImageFileName);
            if (File.Exists(imagePath))
            {
                byte[] imageBytes = null;
                yield return TaskYield.Wait(File.ReadAllBytesAsync(imagePath), b => imageBytes = b,
                    $"Avatar.Load read {imagePath}");
                image = new Texture2D(2, 2);
                if (!image.LoadImage(imageBytes))
                {
                    UnityEngine.Object.DestroyImmediate(image);
                    throw new InvalidDataException($"Could not decode avatar image: {imagePath}");
                }
                image.name = $"avatar_{id}";
            }
            else
            {
                Logger.LogWarning($"[Avatar] No {ImageFileName} in {folder}");
            }

            // Mode: from the manifest when there is one, otherwise inferred.
            CreationMode mode;
            AvatarManifest manifest = null;
            string manifestPath = Path.Combine(folder, AvatarManifest.FileName);
            if (File.Exists(manifestPath))
            {
                string manifestJson = null;
                yield return TaskYield.Wait(File.ReadAllTextAsync(manifestPath), t => manifestJson = t,
                    $"Avatar.Load read {manifestPath}");
                manifest = JsonConvert.DeserializeObject<AvatarManifest>(manifestJson)
                           ?? throw new InvalidDataException($"Could not parse {manifestPath}");
                mode = manifest.mode;
            }
            else if (modeHint.HasValue)
            {
                mode = modeHint.Value;
            }
            else
            {
                string drivingFrames = Path.Combine(folder, DrivingFramesFolderName);
                int count = Directory.Exists(drivingFrames) ? Directory.GetDirectories(drivingFrames).Length : 0;
                mode = count == 0 ? CreationMode.VoiceOnly
                     : count == 1 ? CreationMode.SingleExpression
                     : CreationMode.AllExpressions;
            }

            var avatar = new Avatar(id, mode, folder, image, isLegacy);
            if (manifest != null)
            {
                // A folder from before motion editing has neither field: its
                // frames are one per driving frame and have to be ping-ponged.
                avatar.FrameRate = manifest.fps is > 0f ? manifest.fps.Value : DefaultFrameRate;
                avatar.IsLoopable = manifest.loopable ?? false;
            }
            yield return avatar.LoadExpressionsData();

            Logger.LogVerbose($"[Avatar] Loaded avatar {id} ({avatar.LoadedExpressions.Count} expression(s), " +
                              $"{avatar.FrameRate:F0} fps, {(avatar.IsLoopable ? "loopable" : "ping-pong")}) in {start.Elapsed.TotalMilliseconds:F0}ms");
            onComplete?.Invoke(avatar);
        }

        /// <summary>
        /// Load all expression data (frames, latents, face data)
        /// </summary>
        private IEnumerator LoadExpressionsData()
        {
            string drivingFramesFolder = DrivingFramesFolder;
            if (!Directory.Exists(drivingFramesFolder))
            {
                Logger.LogVerbose($"[Avatar] No driving frames folder: {drivingFramesFolder} (image-only avatar)");
                yield break;
            }

            var expressionFolders = Directory.GetDirectories(drivingFramesFolder);
            Logger.LogVerbose($"[Avatar] Found {expressionFolders.Length} expression folders");

            for (int i = 0; i < expressionFolders.Length; i++)
            {
                string expressionFolder = expressionFolders[i];
                string folderName = Path.GetFileName(expressionFolder);

                // Extract expression index from folder name (expression-0, expression-1, etc.)
                if (folderName.StartsWith("expression-") && int.TryParse(folderName[11..], out int expressionIndex))
                {
                    var expressionData = new ExpressionData
                    {
                        ExpressionName = GetExpressionName(expressionIndex)
                    };
                    // The lip-sync generator walks the frames the same way the
                    // player does: forward with wrap for a loopable set,
                    // ping-pong for a legacy one.
                    expressionData.Data.Loopable = IsLoopable;

                    // Load latents
                    yield return LoadExpressionLatents(expressionFolder, expressionData);

                    // Load face data
                    yield return LoadExpressionFaceData(expressionFolder, expressionData);

                    LoadedExpressions[expressionIndex] = expressionData;
                    Logger.LogVerbose($"[Avatar] Loaded expression {expressionIndex} ({expressionData.ExpressionName}): {expressionData.Data.FaceRegions.Count} frames");
                }
            }
        }

        /// <summary>
        /// Load latents for a specific expression - optimized with unsafe code and parallelization
        /// </summary>
        private static IEnumerator LoadExpressionLatents(string expressionFolder, ExpressionData expressionData)
        {
            string latentsFile = Path.Combine(expressionFolder, "latents.bin");
            if (!File.Exists(latentsFile))
            {
                throw new FileNotFoundException(
                    $"Avatar expression has no latents: {latentsFile}. The avatar did not finish building; recreate it.",
                    latentsFile);
            }

            // Both steps rethrow on fault. Skipping them used to leave the
            // expression with zero latents, which surfaced much later as
            // "No avatar latents available" on the first animated line.
            byte[] latentsBytes = null;
            yield return TaskYield.Wait(File.ReadAllBytesAsync(latentsFile), b => latentsBytes = b,
                $"Avatar.LoadExpressionLatents read {latentsFile}");

            // Process latents in parallel using unsafe code for optimal performance
            yield return TaskYield.Wait(Task.Run(() => ProcessLatentsUnsafe(latentsBytes, expressionData)),
                $"Avatar.LoadExpressionLatents process {latentsFile}");
        }

        /// <summary>
        /// Process latents using unsafe code and parallel processing for maximum performance
        /// </summary>
        private static unsafe void ProcessLatentsUnsafe(byte[] latentsBytes, ExpressionData expressionData)
        {
            const int latentSize = 8 * 32 * 32; // 8192 floats per latent
            const int floatSize = sizeof(float);
            const int latentSizeBytes = latentSize * floatSize;

            int totalFloats = latentsBytes.Length / floatSize;
            int numLatents = totalFloats / latentSize;

            if (numLatents == 0)
            {
                throw new InvalidDataException("latents.bin holds no complete latent.");
            }

            // Pre-allocate list capacity to avoid resizing
            expressionData.Data.Latents.Capacity = numLatents;

            // Create all latent arrays upfront to avoid allocations in parallel loop
            var latentArrays = new float[numLatents][];
            for (int i = 0; i < numLatents; i++)
            {
                latentArrays[i] = new float[latentSize];
            }

            // Pin the source bytes for unsafe access
            fixed (byte* sourcePtr = latentsBytes)
            {
                float* floatPtr = (float*)sourcePtr;

                // Process latents in parallel with optimal memory access
                Parallel.For(0, numLatents, new ParallelOptions
                {
                    MaxDegreeOfParallelism = Environment.ProcessorCount
                }, latentIndex =>
                {
                    var targetArray = latentArrays[latentIndex];
                    float* sourceLatentPtr = floatPtr + (latentIndex * latentSize);

                    // Pin target array for direct memory copy
                    fixed (float* targetPtr = targetArray)
                    {
                        // Direct memory copy - much faster than Array.Copy or Buffer.BlockCopy
                        Buffer.MemoryCopy(sourceLatentPtr, targetPtr, latentSizeBytes, latentSizeBytes);
                    }
                });
            }

            // Add all processed latents to the expression data
            // This is done sequentially to avoid thread safety issues with List<T>
            for (int i = 0; i < numLatents; i++)
            {
                expressionData.Data.Latents.Add(latentArrays[i]);
            }
        }

        /// <summary>
        /// Load face data for a specific expression
        /// </summary>
        private static IEnumerator LoadExpressionFaceData(string expressionFolder, ExpressionData expressionData)
        {
            string facesFile = Path.Combine(expressionFolder, "faces.json");
            if (!File.Exists(facesFile))
            {
                throw new FileNotFoundException(
                    $"Avatar expression has no face data: {facesFile}. The avatar did not finish building; recreate it.",
                    facesFile);
            }

            string facesJson = null;
            yield return TaskYield.Wait(File.ReadAllTextAsync(facesFile), t => facesJson = t,
                $"Avatar.LoadExpressionFaceData read {facesFile}");

            yield return TaskYield.Wait(ParseFaceDataJson(facesJson, expressionData, expressionFolder),
                $"Avatar.LoadExpressionFaceData parse {facesFile}");
        }

        /// <summary>
        /// Parse face data JSON and load all associated textures
        /// </summary>
        private static async Task ParseFaceDataJson(string facesJson, ExpressionData expressionData, string expressionFolder)
        {
            // Parse the JSON using a proper data structure instead of dynamic
            var faceDataJson = JsonConvert.DeserializeObject<FaceDataContainer>(facesJson)
                ?? throw new InvalidDataException($"Could not parse face data in {expressionFolder}");

            if (faceDataJson.faceRegions == null)
                return;

            var tasks = new List<Task>();
            for (int i = 0; i < faceDataJson.faceRegions.Length; i++)
            {
                var faceRegion = faceDataJson.faceRegions[i];
                // Create complete face data structure with all loaded textures
                var faceData = new FaceData
                {
                    HasFace = faceRegion.hasFace,
                    BoundingBox = new Rect(
                        faceRegion.boundingBox.x,
                        faceRegion.boundingBox.y,
                        faceRegion.boundingBox.width,
                        faceRegion.boundingBox.height
                    ),
                    AdjustedFaceBbox = new Vector4(
                        faceRegion.adjustedFaceBbox.x,
                        faceRegion.adjustedFaceBbox.y,
                        faceRegion.adjustedFaceBbox.z,
                        faceRegion.adjustedFaceBbox.w
                    ),
                    CropBox = new Vector4(
                        faceRegion.cropBox.x,
                        faceRegion.cropBox.y,
                        faceRegion.cropBox.z,
                        faceRegion.cropBox.w
                    )
                };
                expressionData.Data.FaceRegions.Add(faceData);
                tasks.Add(LoadFaceTextures(faceData, faceRegion, expressionFolder));
            }
            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Load all face textures from saved files
        /// </summary>
        private static async Task LoadFaceTextures(FaceData faceData, FaceRegionData faceRegion, string expressionFolder)
        {
            // Define texture mappings to eliminate code duplication
            var textureLoaders = new[]
            {
                new {
                    FilePath = faceRegion.textureFiles?.croppedFace,
                    Dimensions = faceRegion.textureDimensions.croppedFace,
                    SetTexture = new Action<Frame>(frame => faceData.CroppedFaceTexture = frame)
                },
                new {
                    FilePath = faceRegion.textureFiles?.faceLarge,
                    Dimensions = faceRegion.textureDimensions.faceLarge,
                    SetTexture = new Action<Frame>(frame => faceData.FaceLarge = frame)
                },
                new {
                    FilePath = faceRegion.textureFiles?.segmentationMask,
                    Dimensions = faceRegion.textureDimensions.segmentationMask,
                    SetTexture = new Action<Frame>(frame => faceData.SegmentationMask = frame)
                },
                new {
                    FilePath = faceRegion.textureFiles?.maskSmall,
                    Dimensions = faceRegion.textureDimensions.maskSmall,
                    SetTexture = new Action<Frame>(frame => faceData.MaskSmall = frame)
                },
                new {
                    FilePath = faceRegion.textureFiles?.original,
                    Dimensions = faceRegion.textureDimensions.original,
                    SetTexture = new Action<Frame>(frame => faceData.OriginalTexture = frame)
                },
                new {
                    FilePath = faceRegion.textureFiles?.fullMask,
                    Dimensions = faceRegion.textureDimensions.fullMask,
                    SetTexture = new Action<Frame>(frame => faceData.FullMask = frame)
                },
                new {
                    FilePath = faceRegion.textureFiles?.boundaryMask,
                    Dimensions = faceRegion.textureDimensions.boundaryMask,
                    SetTexture = new Action<Frame>(frame => faceData.BoundaryMask = frame)
                },
                new {
                    FilePath = faceRegion.textureFiles?.blurredMask,
                    Dimensions = faceRegion.textureDimensions.blurredMask,
                    SetTexture = new Action<Frame>(frame => faceData.BlurredMask = frame)
                }
            };

            var tasks = textureLoaders
                .Where(loader => !string.IsNullOrEmpty(loader.FilePath))
                .Select(loader => Task.Run(async () =>
                {
                    string texturePath = Path.Combine(expressionFolder, loader.FilePath);
                    var frame = await LoadTextureAsFrame(texturePath, loader.Dimensions.width, loader.Dimensions.height);
                    loader.SetTexture(frame);
                }))
                .ToList();

            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Load a texture file and convert it to Frame format
        /// </summary>
        private static async Task<Frame> LoadTextureAsFrame(string texturePath, int width, int height)
        {
            if (!File.Exists(texturePath))
            {
                Logger.LogWarning($"[Avatar] Texture file not found: {texturePath}");
                return new Frame(); // Return empty frame
            }

            return new Frame(await File.ReadAllBytesAsync(texturePath), width, height);
        }

        #endregion
    }
}
