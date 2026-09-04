using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace LiveTalk.API
{
    using Utils;
    /// <summary>
    /// Playback state of a <see cref="CharacterPlayer"/>.
    ///
    /// <code>
    /// Uninitialized → Loading → Ready ⇄ Speaking
    ///                                 ↘ Paused ↗
    /// </code>
    ///
    /// <list type="bullet">
    /// <item><b>Uninitialized</b> — no character, or the assigned character failed to load.</item>
    /// <item><b>Loading</b> — character data and/or idle frames are being loaded. Speech queued now is held and drained on <see cref="CharacterPlayer.OnReady"/>.</item>
    /// <item><b>Ready</b> — character data <i>and</i> idle frames are loaded; idle animation is playing (when enabled) and nothing is queued or in flight. Formerly named <see cref="Idle"/>.</item>
    /// <item><b>Speaking</b> — at least one line is being generated or played. The player stays here across consecutive lines, including lines queued while the last segment was playing.</item>
    /// <item><b>Paused</b> — <see cref="CharacterPlayer.Pause"/> was called. Whether <see cref="CharacterPlayer.Resume"/> returns to Ready or Speaking depends on which state was paused.</item>
    /// </list>
    /// </summary>
    public enum PlaybackState
    {
        Uninitialized,
        Loading,
        Ready,
        Speaking,
        Paused,

        /// <summary>
        /// Same value as <see cref="Ready"/>; kept so existing comparisons compile.
        /// The name was misleading — it read as "idle animation is playing", but it
        /// always meant "loaded and not speaking".
        /// </summary>
        [Obsolete("Use PlaybackState.Ready. Idle and Ready are the same value.")]
        Idle = Ready
    }

    /// <summary>
    /// Speech request data structure
    /// </summary>
    internal class SpeechRequest
    {
        public List<string> TextLines { get; set; } = new List<string>();
        public int ExpressionIndex { get; set; }
        public bool WithAnimation { get; set; } = true;
    }

    /// <summary>
    /// CharacterPlayer is a reusable MonoBehaviour component that handles character loading,
    /// idle animation playback, and speech animation with smooth transitions.
    /// 
    /// Usage:
    /// 1. Add to GameObject with RawImage component
    /// 2. Assign a LiveTalk Character
    /// 3. Call QueueSpeech() to make character speak
    /// 
    /// Features:
    /// - Auto-loads character when assigned
    /// - Plays idle animation (expression 0) at 25 FPS with ping-pong cycling
    /// - Queues and plays speech with smooth transitions
    /// - Seamlessly returns to idle after speech
    /// - Speech may be queued at any time after <see cref="AssignCharacter"/>; lines queued
    ///   before the player is <see cref="PlaybackState.Ready"/> play as soon as it is.
    /// </summary>
    public class CharacterPlayer : MonoBehaviour
    {
        // Inspector-assignable
        [SerializeField] private Texture _displayImage = null;
        [SerializeField] private readonly bool autoPlayIdle = true;
        [SerializeField] private readonly float idleFPS = 25f;
        [SerializeField] private bool audioOnly = false; // For characters without avatars
        
        // Runtime state
        private Character _character;
        public Texture DisplayImage 
        { 
            get 
            { 
                return _displayImage;
            } 
            private set 
            { 
                if (_displayImage != value) 
                {
                    _displayImage = value;
                    OnFrameUpdate?.Invoke(_displayImage);
                }
            }
        }
        private PlaybackState _state = PlaybackState.Uninitialized;
        private readonly Queue<SpeechRequest> _speechQueue = new();

        /// <summary>
        /// Bumped by <see cref="Stop"/>. Every long-running coroutine captures the
        /// value it started under and stops touching player state once it no
        /// longer matches. This is what lets <see cref="Stop"/> retire the speech
        /// processor without killing it mid-<c>SpeakAsync</c> (see the comment in
        /// <see cref="Stop"/> for why that matters).
        /// </summary>
        private int _epoch;

        // Loading
        private Coroutine _loadCoroutine;
        private bool _idleLoaded;
        
        // Idle animation
        private List<Texture> _idleFrames;
        private int _idleFrameIndex = 0;
        private bool _idleForward = true;
        private Coroutine _idleCoroutine;
        
        // Speech animation
        private Coroutine _speechCoroutine;
        private Coroutine _animationCoroutine;
        private AudioSource _audioSource;

        // Pause bookkeeping
        private bool _pausedWhileSpeaking;
        private bool _idleWasRunningAtPause;
        private bool _destroyed;
        
        // Pipelined processing
        private readonly Queue<PendingSpeechItem> _pendingAnimations = new();
        private readonly List<FrameCollector> _frameCollectors = new();
        private bool _isSpeechProcessorRunning = false;
        private bool _isAnimationPlayerRunning = false;
        
        private class PendingSpeechItem
        {
            public List<Texture> Frames { get; set; }
            public AudioClip AudioClip { get; set; }
            public bool AudioReady { get; set; }
            public bool AnimationReady { get; set; }
            public bool IsReady => AudioReady && AnimationReady;
            public FrameStream FrameStream { get; set; }
            public bool WithAnimation { get; set; } = true;
        }

        /// <summary>One in-flight <see cref="CollectAnimationFrames"/>, so <see cref="Stop"/> can find it.</summary>
        private class FrameCollector
        {
            public Coroutine Handle;
        }
        
        // Events
        public event Action<Texture> OnFrameUpdate;
        public event Action OnSpeechStarted;
        public event Action OnSpeechEnded;
        public event Action<Exception> OnError;
        public event Action OnCharacterLoaded;
        public event Action OnIdleStarted;

        /// <summary>
        /// Raised once the assigned character's data <i>and</i> its idle frames are
        /// loaded — the moment the player enters <see cref="PlaybackState.Ready"/>
        /// (or would have, had speech not already been queued: any lines queued
        /// while loading start draining immediately after this event, so a host
        /// does not need to wait for it before calling <see cref="QueueSpeech"/>).
        /// <see cref="OnCharacterLoaded"/> fires right after this, for compatibility.
        /// </summary>
        public event Action OnReady;
        
        // Public properties
        public PlaybackState State => _state;
        public Character Character => _character;

        private static Transform s_parentTransform;

        /// <summary>
        /// Shared scene parent for player GameObjects. Cached after the first
        /// lookup; re-found (or re-created) only if the cached object has been
        /// destroyed, e.g. by a scene change.
        /// </summary>
        public static Transform ParentTransform {
            get {
                // UnityEngine.Object's == null is true for a destroyed object too.
                if (s_parentTransform == null)
                {
                    GameObject parent = GameObject.Find("CharacterPlayers_Parent");
                    if (parent == null)
                    {
                        parent = new GameObject("CharacterPlayers_Parent");
                        parent.transform.SetParent(null); // Keep at root for persistence
                    }
                    s_parentTransform = parent.transform;
                }
                return s_parentTransform;
            }
        }

        /// <summary>
        /// True only while the player is <see cref="PlaybackState.Speaking"/>: a
        /// line is being generated or played. False while Ready (idle animation
        /// does not count as playing), Loading, Paused or Uninitialized. A host
        /// that wants "nothing left to say" should test
        /// <c>!IsPlaying &amp;&amp; QueuedSpeechCount == 0</c>, or listen for
        /// <see cref="OnSpeechEnded"/>, which fires exactly when that becomes true.
        /// </summary>
        public bool IsPlaying => _state == PlaybackState.Speaking;

        /// <summary>
        /// True when character data and idle frames are loaded and nothing is
        /// speaking, queued or paused — i.e. <see cref="State"/> is
        /// <see cref="PlaybackState.Ready"/>.
        /// </summary>
        public bool IsReady => _state == PlaybackState.Ready;

        public int QueuedSpeechCount => _speechQueue.Count;

        private void Awake()
        {
            // Ensure we have an audio source
            _audioSource = GetComponent<AudioSource>();
            if (_audioSource == null)
            {
                _audioSource = gameObject.AddComponent<AudioSource>();
            }
            _audioSource.playOnAwake = false;
        }

        /// <summary>
        /// Assign a character to this player. If character is not loaded, it will be loaded automatically.
        /// If the same character is already assigned (and loading or loaded), this is a no-op.
        /// The player is <see cref="PlaybackState.Loading"/> until both the character data and
        /// its idle frames are in, then raises <see cref="OnReady"/>.
        /// </summary>
        public void AssignCharacter(Character character)
        {
            if (_character == character && character != null && _state != PlaybackState.Uninitialized)
            {
                // Same character, already loading or loaded.
                return;
            }
            
            // Stop current playback. A load of the previous character must not
            // finish into this one, so it is stopped here rather than in Stop():
            // Stop() itself deliberately leaves a load running (see there).
            Stop();
            // Stop() put the previous character's idle back on screen; the new
            // one is Loading, which shows nothing until its own frames are in.
            StopIdleAnimation();
            if (_loadCoroutine != null)
            {
                StopCoroutine(_loadCoroutine);
                _loadCoroutine = null;
            }
            _idleLoaded = false;
            
            _character = character;
            
            if (character == null)
            {
                _state = PlaybackState.Uninitialized;
                return;
            }

            _state = PlaybackState.Loading;
            _loadCoroutine = StartCoroutine(LoadCharacterCoroutine(character));
        }

        /// <summary>
        /// Queue speech for the character. Speech is played in order; text is
        /// broken into lines/sentences for smoother playback.
        ///
        /// Guarantees:
        /// <list type="bullet">
        /// <item>May be called as soon as a character is assigned. While the player is
        /// <see cref="PlaybackState.Loading"/> the request is only queued and drains
        /// automatically when <see cref="OnReady"/> fires — no host-side wait is needed.</item>
        /// <item>While <see cref="PlaybackState.Speaking"/>, the request joins the current
        /// run: generation of the new line starts right away and the player stays Speaking
        /// through it (no intermediate <see cref="OnSpeechEnded"/> / <see cref="OnSpeechStarted"/>).</item>
        /// <item>While <see cref="PlaybackState.Paused"/> the request is queued and starts on
        /// <see cref="Resume"/>.</item>
        /// <item>Dropped, with a warning, only when there is no character or the assigned
        /// character failed to load (<see cref="PlaybackState.Uninitialized"/>), or when the
        /// text is empty.</item>
        /// </list>
        /// </summary>
        /// <param name="withAnimation">If false, plays audio only (useful for characters without avatars)</param>
        public void QueueSpeech(string text, int expressionIndex = 0, bool withAnimation = true)
        {
            if (_character == null)
            {
                Logger.LogWarning("[CharacterPlayer] Cannot queue speech: no character assigned");
                return;
            }

            if (_state == PlaybackState.Uninitialized)
            {
                // A character is assigned but its load failed (a load in progress
                // is Loading, not Uninitialized). Nothing will ever drain the queue.
                Logger.LogWarning($"[CharacterPlayer] Cannot queue speech: character {_character.Name} did not load");
                return;
            }
            
            if (string.IsNullOrEmpty(text))
            {
                Logger.LogWarning("[CharacterPlayer] Cannot queue empty speech");
                return;
            }
            
            // Override animation if in audio-only mode or explicitly disabled
            if (audioOnly)
                withAnimation = false;
            
            // Break text into lines using TextUtils (same as SpeechPlaybackManager)
            var lines = TextUtils.BreakTextIntoLines(text);
            if (lines.Length == 0)
            {
                Logger.LogWarning("[CharacterPlayer] No lines after text processing");
                return;
            }
            
            var request = new SpeechRequest 
            { 
                TextLines = new List<string>(lines),
                ExpressionIndex = expressionIndex,
                WithAnimation = withAnimation
            };
            
            _speechQueue.Enqueue(request);
            
            Logger.Log($"[CharacterPlayer] Queued speech: {lines.Length} lines from text: {text.Substring(0, Math.Min(50, text.Length))}... (Animation: {withAnimation}, State: {_state})");

            switch (_state)
            {
                case PlaybackState.Ready:
                case PlaybackState.Speaking:
                    // Ready: start. Speaking: make sure the processor is running —
                    // it exits when the queue empties, so a line queued during the
                    // last segment would otherwise sit until the player loop ends.
                    ProcessNextSpeech();
                    break;
                case PlaybackState.Loading:
                    Logger.Log("[CharacterPlayer] Character still loading — speech will start on Ready");
                    break;
                case PlaybackState.Paused:
                    Logger.Log("[CharacterPlayer] Player paused — speech will start on Resume");
                    break;
            }
        }

        /// <summary>
        /// Stop all playback and clear queues. Idle animation, the speech processor,
        /// the segment player and every in-flight frame collector are stopped; both
        /// queues are cleared. A character load that is still running is left to
        /// finish (it is not playback) and the player returns to
        /// <see cref="PlaybackState.Ready"/> once loaded, otherwise stays
        /// <see cref="PlaybackState.Loading"/>. With no character: Uninitialized.
        /// </summary>
        public void Stop()
        {
            // Retire every running coroutine first: anything still suspended sees
            // a stale epoch when it resumes and exits without touching state.
            _epoch++;

            StopIdleAnimation();

            if (_animationCoroutine != null)
            {
                StopCoroutine(_animationCoroutine);
                _animationCoroutine = null;
            }

            // The speech processor is *retired*, not killed. While it is suspended
            // inside Character.SpeakAsync the VoiceQueue lease is held by that
            // nested coroutine, and Unity documents nothing about running a
            // stopped iterator's finally blocks — least of all a nested one's.
            // Instead the processor checks the epoch after each SpeakAsync returns
            // and exits; the in-flight synthesis completes on its normal path,
            // which is the one place the lease is guaranteed to be released. The
            // result is discarded. MuseTalk generation runs on the API's own
            // controller and is never started or stopped from here, so its lease
            // is not at risk either.
            _speechCoroutine = null;

            // Copy first: if Unity does dispose a stopped iterator, the
            // collector's finally removes itself from this list.
            var collectors = _frameCollectors.ToArray();
            _frameCollectors.Clear();
            foreach (var collector in collectors)
            {
                if (collector.Handle != null)
                    StopCoroutine(collector.Handle);
            }
            
            if (_audioSource != null)
            {
                // Stop() rather than "if isPlaying": a paused source reports
                // isPlaying == false and would otherwise keep its clip.
                _audioSource.Stop();
                _audioSource.clip = null;
            }
            
            _speechQueue.Clear();
            _pendingAnimations.Clear();

            // Reset processing flags so new speech can start fresh. Leaving
            // them set survives the stop: AnimationPlayerLoop is gated on
            // _isAnimationPlayerRunning, so the next request never starts its
            // player loop — audio plays and frames are generated, but none of
            // them are ever displayed, which reads as "the voice works and
            // lip-sync doesn't".
            _isSpeechProcessorRunning = false;
            _isAnimationPlayerRunning = false;
            _pausedWhileSpeaking = false;
            _idleWasRunningAtPause = false;

            if (_character == null)
            {
                _state = PlaybackState.Uninitialized;
            }
            else if (_idleLoaded)
            {
                _state = PlaybackState.Ready;
                if (!_destroyed && autoPlayIdle && !audioOnly && _idleFrames != null && _idleFrames.Count > 0)
                {
                    _idleFrameIndex = 0;
                    _idleForward = true;
                    StartIdleAnimation();
                }
            }
            else if (_loadCoroutine != null)
            {
                _state = PlaybackState.Loading;
            }
            else
            {
                _state = PlaybackState.Uninitialized;
            }
        }

        /// <summary>
        /// Pause playback: idle animation stops, and if a line was playing its
        /// audio is paused and frame playback holds. No-op unless the player is
        /// Ready or Speaking. Speech queued while paused starts on <see cref="Resume"/>.
        /// </summary>
        public void Pause()
        {
            if (_state != PlaybackState.Ready && _state != PlaybackState.Speaking)
                return;

            _pausedWhileSpeaking = _state == PlaybackState.Speaking;
            _idleWasRunningAtPause = _idleCoroutine != null;

            if (_pausedWhileSpeaking && _audioSource != null)
            {
                _audioSource.Pause();
            }
            StopIdleAnimation();
            _state = PlaybackState.Paused;
        }

        /// <summary>
        /// Resume playback. Returns to <see cref="PlaybackState.Speaking"/> only if
        /// the player was speaking when paused; otherwise to
        /// <see cref="PlaybackState.Ready"/> with idle animation. Anything queued
        /// while paused then starts.
        /// </summary>
        public void Resume()
        {
            if (_state != PlaybackState.Paused)
                return;

            if (_pausedWhileSpeaking)
            {
                _state = PlaybackState.Speaking;
                if (_audioSource != null)
                    _audioSource.UnPause();
                // Paused between segments: the loop was idling while it waited
                // for the next one, so put the idle back.
                if (_idleWasRunningAtPause)
                    StartIdleAnimation();
            }
            else
            {
                _state = PlaybackState.Ready;
                if (autoPlayIdle && !audioOnly && _idleFrames != null && _idleFrames.Count > 0)
                    StartIdleAnimation();
            }

            _pausedWhileSpeaking = false;
            _idleWasRunningAtPause = false;

            if (_speechQueue.Count > 0)
                ProcessNextSpeech();
        }

        /// <summary>
        /// Clear all queued speech
        /// </summary>
        public void ClearQueue()
        {
            _speechQueue.Clear();
        }

        /// <summary>
        /// Loads character data if needed, then idle frames, then moves to Ready.
        /// Never writes the state while a speech is in flight (it cannot be — speech
        /// queued while Loading is only queued — but the guard is what the contract
        /// promises), and drains anything queued meanwhile.
        /// </summary>
        private IEnumerator LoadCharacterCoroutine(Character character)
        {
            if (!character.IsDataLoaded)
            {
                Logger.Log($"[CharacterPlayer] Loading character: {character.Name}");

                Exception loadError = null;
                yield return TaskYield.Guard(character.LoadData(),
                    ex => loadError = ex, "CharacterPlayer.LoadCharacter");

                if (loadError != null || !character.IsDataLoaded)
                {
                    loadError ??= new InvalidOperationException(
                        $"Character {character.Name} did not finish loading.");
                    Logger.LogError($"[CharacterPlayer] Failed to load character {character.Name}: {loadError.Message}");
                    _loadCoroutine = null;
                    _state = PlaybackState.Uninitialized;
                    _speechQueue.Clear();
                    OnError?.Invoke(loadError);
                    yield break;
                }

                Logger.Log($"[CharacterPlayer] Character loaded successfully: {character.Name}");
            }

            // Load idle frames with yielding
            yield return LoadIdleFramesCoroutine();
            
            // If no idle frames and not audio-only, use static character image as fallback
            if (_idleFrames == null || _idleFrames.Count == 0)
            {
                if (!audioOnly && _character?.Image != null)
                {
                    Logger.Log($"[CharacterPlayer] No idle frames loaded for {_character?.Name}. Using static character image.");
                    DisplayImage = _character.Image;
                    audioOnly = true; // Switch to audio-only mode for speech
                }
                else if (!audioOnly)
                {
                    Logger.LogWarning($"[CharacterPlayer] No idle frames or static image available for {_character?.Name}. Using audio-only mode.");
                    audioOnly = true;
                }
                else
                {
                    Logger.Log($"[CharacterPlayer] Audio-only mode for {_character?.Name}");
                }
            }

            _idleLoaded = true;
            _loadCoroutine = null;

            // Ready means "loaded and nothing in flight". If something is in
            // flight, leave the state alone: writing Ready/Idle here used to
            // land mid-speech and make PlayFramesSynchronized exit on entry.
            bool speechInFlight = _state == PlaybackState.Speaking
                                  || _isSpeechProcessorRunning
                                  || _isAnimationPlayerRunning;
            if (!speechInFlight)
            {
                _state = PlaybackState.Ready;
                if (autoPlayIdle && !audioOnly && _idleFrames != null && _idleFrames.Count > 0)
                {
                    StartIdleAnimation();
                }
            }
            
            // Fire events AFTER idle frames are loaded and state is set
            OnReady?.Invoke();
            OnCharacterLoaded?.Invoke();

            // Drain whatever was queued while loading. Idle keeps playing until
            // the first segment is ready, as it does for any speech.
            if (_state == PlaybackState.Ready && _speechQueue.Count > 0)
            {
                Logger.Log($"[CharacterPlayer] Ready — starting {_speechQueue.Count} request(s) queued while loading");
                ProcessNextSpeech();
            }
        }

        private IEnumerator LoadIdleFramesCoroutine()
        {
            _idleFrames = new List<Texture>();
            
            if (_character == null || string.IsNullOrEmpty(_character.CharacterFolder))
            {
                Logger.LogWarning($"[CharacterPlayer] Cannot load idle frames: character or folder is null");
                yield break;
            }
            
            // Expression 0 folder path
            string expression0Folder = Path.Combine(_character.CharacterFolder, "drivingFrames", "expression-0");
            
            if (!Directory.Exists(expression0Folder))
            {
                Logger.LogWarning($"[CharacterPlayer] Expression 0 folder not found: {expression0Folder}");
                yield break;
            }
            
            // Load all PNG frames from the expression folder
            var framePaths = Directory.GetFiles(expression0Folder, "*.png")
                .OrderBy(p => p)
                .ToArray();
            
            if (framePaths.Length == 0)
            {
                Logger.LogWarning($"[CharacterPlayer] No frames found in: {expression0Folder}");
                yield break;
            }
            
            // Load textures from disk with yielding between each
            foreach (var framePath in framePaths)
            {
                // Read file asynchronously. A single unreadable frame is skipped
                // (observed and logged), not fatal to the whole idle set.
                var readTask = File.ReadAllBytesAsync(framePath);
                yield return new WaitUntil(() => readTask.IsCompleted);
                
                if (readTask.IsCompletedSuccessfully)
                {
                    byte[] fileData = readTask.Result;
                    Texture2D texture = new(2, 2);
                    if (texture.LoadImage(fileData))
                    {
                        _idleFrames.Add(texture);
                    }
                }
                else if (readTask.IsFaulted)
                {
                    Logger.LogError($"[CharacterPlayer] Failed to read frame {framePath}: {readTask.Exception?.GetBaseException().Message}");
                }
                
                // Yield after each texture to avoid blocking
                yield return null;
            }
            
            if (_idleFrames.Count == 0)
            {
                Logger.LogWarning($"[CharacterPlayer] No frames loaded for character: {_character.Name}");
            }
            else
            {
                Logger.Log($"[CharacterPlayer] Loaded {_idleFrames.Count} idle frames from {expression0Folder}");
            }
        }

        private void StartIdleAnimation()
        {
            if (_idleCoroutine != null)
            {
                StopCoroutine(_idleCoroutine);
            }
            
            _idleFrameIndex = 0;
            _idleForward = true;
            _idleCoroutine = StartCoroutine(PlayIdleAnimation());
            OnIdleStarted?.Invoke();
        }

        private void StopIdleAnimation()
        {
            if (_idleCoroutine != null)
            {
                StopCoroutine(_idleCoroutine);
                _idleCoroutine = null;
            }
        }

        private IEnumerator PlayIdleAnimation()
        {
            float frameInterval = 1f / idleFPS;
            
            // Continue while idle coroutine is running (controlled by Start/Stop)
            // Don't check state - we explicitly start/stop this coroutine
            while (true)
            {
                if (_idleFrames == null || _idleFrames.Count == 0)
                {
                    yield return new WaitForSeconds(0.1f);
                    continue;
                }
                
                // Display current frame
                Texture currentFrame = _idleFrames[_idleFrameIndex];
                DisplayImage = currentFrame;
                
                // Advance frame index with ping-pong logic (no duplicate frames)
                if (_idleForward)
                {
                    _idleFrameIndex++;
                    if (_idleFrameIndex >= _idleFrames.Count)
                    {
                        // Reached end, go reverse (skip last frame to avoid duplicate)
                        _idleFrameIndex = Math.Max(0, _idleFrames.Count - 2);
                        _idleForward = false;
                    }
                }
                else // Going reverse
                {
                    _idleFrameIndex--;
                    if (_idleFrameIndex < 0)
                    {
                        // Reached start, go forward (skip first frame to avoid duplicate)
                        _idleFrameIndex = Math.Min(1, _idleFrames.Count - 1);
                        _idleForward = true;
                    }
                }
                
                yield return new WaitForSeconds(frameInterval);
            }
        }

        /// <summary>
        /// Ensures both pipeline loops are running for whatever is queued. Idempotent:
        /// the running flags are set <i>before</i> <c>StartCoroutine</c>, so two calls in
        /// the same frame (or a call while a loop is mid-flight) cannot start a second
        /// loop. Enters Speaking (firing <see cref="OnSpeechStarted"/>) only on the
        /// transition; a call while already Speaking just tops up the pipeline.
        /// </summary>
        private void ProcessNextSpeech()
        {
            if (_speechQueue.Count == 0)
                return;

            if (_state != PlaybackState.Speaking)
            {
                _state = PlaybackState.Speaking;
                // DON'T stop idle animation yet - let it play while speech is being generated
                // AnimationPlayerLoop will stop it when first segment is ready to play
                OnSpeechStarted?.Invoke();
            }
            
            // Start both processor and player loops for pipelining. Flags first:
            // StartCoroutine runs the loop to its first yield synchronously, and
            // the player loop's exit condition reads the processor's flag.
            if (!_isSpeechProcessorRunning)
            {
                _isSpeechProcessorRunning = true;
                _speechCoroutine = StartCoroutine(SpeechProcessorLoop(_epoch));
            }
            
            if (!_isAnimationPlayerRunning)
            {
                _isAnimationPlayerRunning = true;
                _animationCoroutine = StartCoroutine(AnimationPlayerLoop(_epoch));
            }
        }

        /// <summary>
        /// Processes queued speech requests, generating audio/animation for each line.
        /// Audio generation is serialized, but animation collection happens in parallel.
        /// Runs in parallel with AnimationPlayerLoop for pipelining.
        /// </summary>
        /// <param name="epoch">
        /// The <see cref="_epoch"/> this loop belongs to. After <see cref="Stop"/> the
        /// loop is not stopped but retired: it lets the SpeakAsync it is inside finish
        /// (so the voice lease is released on its normal path), then exits without
        /// touching state.
        /// </param>
        private IEnumerator SpeechProcessorLoop(int epoch)
        {
            bool completed = false;
            try
            {
                while (epoch == _epoch && _speechQueue.Count > 0)
                {
                    var request = _speechQueue.Dequeue();
                    Logger.Log($"[CharacterPlayer] Processing speech request: {request.TextLines.Count} lines");
                    
                    foreach (var line in request.TextLines)
                    {
                        if (string.IsNullOrWhiteSpace(line))
                            continue;
                        
                        Logger.Log($"[CharacterPlayer] Generating line: {line.Substring(0, Math.Min(50, line.Length))}...");
                        
                        // Create pending item for this line
                        var pendingItem = new PendingSpeechItem
                        {
                            Frames = new List<Texture>(),
                            AudioClip = null,
                            AudioReady = false,
                            AnimationReady = false,
                            FrameStream = null,
                            WithAnimation = request.WithAnimation
                        };
                        
                        _pendingAnimations.Enqueue(pendingItem);
                        
                        // For audio-only mode, generate only audio
                        if (!request.WithAnimation || audioOnly)
                        {
                            // Generate audio only using SpeakAsync with expressionIndex = -1
                            AudioClip audioClip = null;
                            bool hasError = false;
                            
                            IEnumerator audioCoroutine = _character.SpeakAsync(
                                line,
                                expressionIndex: -1, // -1 means audio-only
                                onAudioReady: (stream, clip) =>
                                {
                                    audioClip = clip;
                                    // stream will be null for audio-only
                                },
                                onAnimationComplete: null,
                                onError: (ex) =>
                                {
                                    hasError = true;
                                    ReportSpeechError(epoch, "Audio generation error", ex);
                                }
                            );
                            
                            yield return audioCoroutine;

                            // Stopped while synthesising: the line is discarded.
                            if (epoch != _epoch)
                                yield break;
                            
                            if (audioClip != null && !hasError)
                            {
                                pendingItem.AudioClip = audioClip;
                                pendingItem.AudioReady = true;
                                pendingItem.AnimationReady = true; // No animation needed
                                Logger.Log($"[CharacterPlayer] Audio-only ready: {audioClip.length}s");
                            }
                            else
                            {
                                Logger.LogWarning($"[CharacterPlayer] Failed to generate audio for line");
                                pendingItem.AudioReady = true;
                                pendingItem.AnimationReady = true;
                            }
                            
                            continue;
                        }
                        
                        // Generate speech audio + animation
                        FrameStream frameStream = null;
                        AudioClip audioClip2 = null;
                        bool hasError2 = false;
                        
                        IEnumerator speechCoroutine = _character.SpeakAsync(
                            line,
                            request.ExpressionIndex,
                            onAudioReady: (stream, clip) =>
                            {
                                frameStream = stream;
                                audioClip2 = clip;
                                pendingItem.AudioClip = clip;
                                pendingItem.FrameStream = stream;
                                pendingItem.AudioReady = true;
                                Logger.Log($"[CharacterPlayer] Audio ready: {clip.length}s - can start next audio generation!");
                            },
                            onAnimationComplete: (stream) =>
                            {
                                Logger.Log($"[CharacterPlayer] Animation generation complete: {stream?.TotalExpectedFrames ?? 0} frames");
                            },
                            onError: (ex) =>
                            {
                                hasError2 = true;
                                ReportSpeechError(epoch, "Speech error", ex);
                            }
                        );
                        
                        // Start the speech generation
                        yield return speechCoroutine;

                        // Stopped while synthesising: the line is discarded. Any
                        // animation the character started for it runs to completion
                        // on the API controller (that is where its lease lives).
                        if (epoch != _epoch)
                            yield break;
                        
                        if (hasError2 || audioClip2 == null || frameStream == null)
                        {
                            Logger.LogWarning($"[CharacterPlayer] Failed to generate speech for line");
                            pendingItem.AudioReady = true;
                            pendingItem.AnimationReady = true; // Mark as ready (but empty) so player can skip
                            continue;
                        }
                        
                        // Audio is ready! Start a separate coroutine to collect frames
                        // in parallel. Tracked so Stop() can find it.
                        var collector = new FrameCollector();
                        _frameCollectors.Add(collector);
                        collector.Handle = StartCoroutine(CollectAnimationFrames(pendingItem, frameStream, collector));
                        
                        // DON'T wait for animation - immediately continue to next line's audio generation
                        Logger.Log($"[CharacterPlayer] Audio done for line - starting next audio generation immediately!");
                    }
                }

                completed = true;
            }
            finally
            {
                // A retired loop (stale epoch) belongs to a previous Stop(); its
                // flags were reset there and may already describe a new loop.
                if (epoch == _epoch)
                {
                    _isSpeechProcessorRunning = false;
                    _speechCoroutine = null;

                    if (!completed)
                    {
                        // Died on an exception: release anything the player loop
                        // is waiting for, or it waits forever on an item that
                        // never becomes ready.
                        foreach (var item in _pendingAnimations)
                        {
                            item.AudioReady = true;
                            item.AnimationReady = true;
                        }
                    }
                }
            }

            Logger.Log("[CharacterPlayer] Speech processor loop ended");
        }

        /// <summary>
        /// Routes a SpeakAsync failure to <see cref="OnError"/> unless the loop that
        /// asked for it has since been retired by <see cref="Stop"/>.
        /// </summary>
        private void ReportSpeechError(int epoch, string what, Exception ex)
        {
            if (epoch != _epoch)
            {
                Logger.LogWarning($"[CharacterPlayer] {what} after Stop() (ignored): {ex.Message}");
                return;
            }
            Logger.LogError($"[CharacterPlayer] {what}: {ex.Message}");
            OnError?.Invoke(ex);
        }
        
        /// <summary>
        /// Collects animation frames in parallel with audio generation for next segment.
        /// </summary>
        private IEnumerator CollectAnimationFrames(PendingSpeechItem item, FrameStream frameStream, FrameCollector self)
        {
            try
            {
                Logger.Log($"[CharacterPlayer] Starting animation frame collection in parallel...");
                
                // Collect all frames. A failed producer marks the stream finished
                // too, so this exits either way; Error tells the two apart.
                while (frameStream.HasMoreFrames)
                {
                    var awaiter = frameStream.WaitForNext();
                    yield return awaiter;
                    
                    if (awaiter.Texture != null)
                    {
                        item.Frames.Add(awaiter.Texture);
                    }
                }

                if (frameStream.Error != null)
                {
                    Logger.LogWarning($"[CharacterPlayer] Animation failed after {item.Frames.Count} frame(s); segment plays with what arrived: {frameStream.Error.Message}");
                }
                else
                {
                    Logger.Log($"[CharacterPlayer] Animation frames collected: {item.Frames.Count} frames");
                }
            }
            finally
            {
                // Mark animation as ready (on every exit: the player loop must
                // never wait on an item nobody is filling).
                item.AnimationReady = true;
                _frameCollectors.Remove(self);
            }
        }

        /// <summary>
        /// Plays generated animation frames synchronized with audio.
        /// Runs in parallel with SpeechProcessorLoop for pipelining.
        /// Returns to idle animation only if waiting for next segment.
        /// Holds while Paused. On completion: restarts the pipeline if more speech
        /// was queued meanwhile, otherwise returns the player to Ready.
        /// </summary>
        private IEnumerator AnimationPlayerLoop(int epoch)
        {
            bool completed = false;
            try
            {
                bool isFirstSegment = true;
                
                while (_isSpeechProcessorRunning || _pendingAnimations.Count > 0)
                {
                    // Check if we need to wait for next segment
                    bool needsToWait = _pendingAnimations.Count == 0 || !_pendingAnimations.Peek().IsReady;
                    
                    if (needsToWait && !isFirstSegment && _state != PlaybackState.Paused)
                    {
                        // Next segment not ready - return to idle while waiting
                        Logger.Log("[CharacterPlayer] Next segment not ready - returning to idle while waiting");
                        _idleFrameIndex = 0;
                        _idleForward = true;
                        StartIdleAnimation();
                    }
                    
                    // Wait for next segment to be ready (idle animates during this wait)
                    while (_pendingAnimations.Count == 0 || !_pendingAnimations.Peek().IsReady)
                    {
                        yield return new WaitForSeconds(0.05f);
                        
                        // Exit if processor finished and no more pending
                        if (!_isSpeechProcessorRunning && _pendingAnimations.Count == 0)
                        {
                            break;
                        }
                    }
                    
                    if (_pendingAnimations.Count == 0)
                        break;
                    
                    var item = _pendingAnimations.Dequeue();

                    // Paused while waiting: hold before touching the display.
                    while (_state == PlaybackState.Paused)
                        yield return null;
                    
                    // Skip empty items
                    if (item.AudioClip == null)
                    {
                        Logger.LogWarning("[CharacterPlayer] Skipping empty speech item");
                        continue;
                    }
                    
                    // For audio-only playback
                    if (!item.WithAnimation || item.Frames.Count == 0)
                    {
                        Logger.Log($"[CharacterPlayer] Playing audio-only: {item.AudioClip.length}s");
                        
                        // Just play the audio
                        _audioSource.clip = item.AudioClip;
                        _audioSource.Play();
                        
                        // Wait for audio to finish (a paused source is not
                        // "playing", so hold on the state as well)
                        while (_state == PlaybackState.Paused || _audioSource.isPlaying)
                        {
                            yield return new WaitForSeconds(0.1f);
                        }
                        
                        // After playing, we're no longer in first segment
                        isFirstSegment = false;
                        continue;
                    }
                    
                    // Stop idle and transition to last idle frame before playing
                    if (needsToWait || isFirstSegment)
                    {
                        // We were in idle (either first time or returned to idle) - do smooth transition
                        Logger.Log("[CharacterPlayer] Segment ready - transitioning from idle to speech");
                        StopIdleAnimation();
                        
                        // Transition idle to its last frame for smooth start
                        if (_idleFrames != null && _idleFrames.Count > 0)
                        {
                            Texture lastIdleFrame = _idleFrames[^1];
                            DisplayImage = lastIdleFrame;
                            
                            // Brief pause to show transition
                            yield return new WaitForSeconds(0.04f);
                        }
                    }
                    else
                    {
                        // Next segment was already ready - play immediately (no idle transition)
                        Logger.Log("[CharacterPlayer] Next segment ready - playing immediately");
                    }
                    
                    Logger.Log($"[CharacterPlayer] Playing segment: {item.Frames.Count} frames, {item.AudioClip.length}s");
                    
                    // Play this segment with its audio
                    yield return PlayFramesSynchronized(item.Frames, item.AudioClip);
                    
                    // After playing, we're no longer in first segment
                    isFirstSegment = false;
                }

                completed = true;
            }
            finally
            {
                // Exception path only. Normal completion continues below (and a
                // stopped coroutine has a stale epoch — Stop() reset the flags).
                if (!completed && epoch == _epoch)
                {
                    _isAnimationPlayerRunning = false;
                    _animationCoroutine = null;
                    if (_state == PlaybackState.Speaking)
                        _state = _idleLoaded ? PlaybackState.Ready : PlaybackState.Loading;
                }
            }

            Logger.Log("[CharacterPlayer] Animation player loop ended");

            // Finish while paused would put idle on screen under a Pause().
            while (_state == PlaybackState.Paused)
                yield return null;

            if (epoch != _epoch)
                yield break;

            _isAnimationPlayerRunning = false;
            _animationCoroutine = null;

            // Lines queued during the last segment: keep going, still Speaking.
            // (QueueSpeech restarts the processor itself; this is the backstop
            // for lines queued while paused, or in the same frame the loop ended.)
            if (_speechQueue.Count > 0)
            {
                Logger.Log($"[CharacterPlayer] {_speechQueue.Count} request(s) still queued - continuing without returning to idle");
                ProcessNextSpeech();
                yield break;
            }

            // Speech complete
            _state = PlaybackState.Ready;
            OnSpeechEnded?.Invoke();
            
            // Return to idle (starting from frame 0 for smooth loop)
            _idleFrameIndex = 0;
            _idleForward = true;
            
            if (autoPlayIdle && !audioOnly && _idleFrames != null && _idleFrames.Count > 0)
            {
                StartIdleAnimation();
            }
        }

        private IEnumerator PlayFramesSynchronized(List<Texture> frames, AudioClip audioClip)
        {
            if (frames.Count == 0 || audioClip == null)
            {
                yield break;
            }
            
            // Calculate frame interval based on audio duration and frame count
            float frameInterval = audioClip.length / frames.Count;
            
            // Start audio playback
            _audioSource.clip = audioClip;
            _audioSource.Play();
            
            // Play frames at calculated rate. Holds while Paused (audio is paused
            // by Pause()); exits only if the player left Speaking some other way.
            for (int i = 0; i < frames.Count; i++)
            {
                while (_state == PlaybackState.Paused)
                    yield return null;
                if (_state != PlaybackState.Speaking)
                    yield break;

                DisplayImage = frames[i];
                yield return new WaitForSeconds(frameInterval);
            }
            
            // Wait for audio to finish
            while (_state == PlaybackState.Paused
                   || (_audioSource.isPlaying && _state == PlaybackState.Speaking))
            {
                yield return null;
            }
        }

        private void OnDestroy()
        {
            // Stop() restarts idle when the character is loaded; not on a
            // GameObject that is going away.
            _destroyed = true;
            Stop();
        }
    }
}
