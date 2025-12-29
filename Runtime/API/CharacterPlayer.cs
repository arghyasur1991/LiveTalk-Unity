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
    /// Playback state of the character player
    /// </summary>
    public enum PlaybackState
    {
        Uninitialized,
        Loading,
        Idle,
        Speaking,
        Paused
    }

    /// <summary>
    /// Speech request data structure
    /// </summary>
    internal class SpeechRequest
    {
        public List<string> TextLines { get; set; } = new List<string>();
        public int ExpressionIndex { get; set; }
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
    /// </summary>
    public class CharacterPlayer : MonoBehaviour
    {
        // Inspector-assignable
        [SerializeField] private RawImage displayImage;
        [SerializeField] private bool autoPlayIdle = true;
        [SerializeField] private float idleFPS = 25f;
        
        // Runtime state
        private Character _character;
        private PlaybackState _state = PlaybackState.Uninitialized;
        private Queue<SpeechRequest> _speechQueue = new Queue<SpeechRequest>();
        
        // Idle animation
        private List<Texture2D> _idleFrames;
        private int _idleFrameIndex = 0;
        private bool _idleForward = true;
        private Coroutine _idleCoroutine;
        
        // Speech animation
        private Coroutine _speechCoroutine;
        private AudioSource _audioSource;
        
        // Pipelined processing
        private Queue<PendingSpeechItem> _pendingAnimations = new Queue<PendingSpeechItem>();
        private bool _isSpeechProcessorRunning = false;
        private bool _isAnimationPlayerRunning = false;
        
        private class PendingSpeechItem
        {
            public List<Texture2D> Frames { get; set; }
            public AudioClip AudioClip { get; set; }
            public bool AudioReady { get; set; }
            public bool AnimationReady { get; set; }
            public bool IsReady => AudioReady && AnimationReady;
            public FrameStream FrameStream { get; set; }
        }
        
        // Events
        public event Action<Texture2D> OnFrameUpdate;
        public event Action OnSpeechStarted;
        public event Action OnSpeechEnded;
        public event Action<Exception> OnError;
        public event Action OnCharacterLoaded;
        public event Action OnIdleStarted;
        
        // Public properties
        public PlaybackState State => _state;
        public Character Character => _character;
        public bool IsPlaying => _state == PlaybackState.Speaking || _state == PlaybackState.Idle;
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
            
            // Auto-find RawImage if not assigned
            if (displayImage == null)
            {
                displayImage = GetComponent<RawImage>();
            }
        }

        /// <summary>
        /// Assign a character to this player. If character is not loaded, it will be loaded automatically.
        /// If the same character is already assigned, this is a no-op.
        /// </summary>
        public void AssignCharacter(Character character)
        {
            if (_character == character && character != null)
            {
                // Same character, check if already loaded
                if (character.IsDataLoaded && _state != PlaybackState.Uninitialized)
                    return; // No-op
            }
            
            // Stop current playback
            Stop();
            
            _character = character;
            
            if (character == null)
            {
                _state = PlaybackState.Uninitialized;
                return;
            }
            
            // Load character if not already loaded
            if (!character.IsDataLoaded)
            {
                _state = PlaybackState.Loading;
                StartCoroutine(LoadCharacterCoroutine(character));
            }
            else
            {
                OnCharacterLoadedInternal();
            }
        }

        /// <summary>
        /// Queue speech for the character. Speech will be played in order.
        /// Text is automatically broken into lines/sentences for smoother playback.
        /// </summary>
        public void QueueSpeech(string text, int expressionIndex = 0)
        {
            if (_character == null || !_character.IsDataLoaded)
            {
                Debug.LogWarning("[CharacterPlayer] Cannot queue speech: Character not loaded");
                return;
            }
            
            if (string.IsNullOrEmpty(text))
            {
                Debug.LogWarning("[CharacterPlayer] Cannot queue empty speech");
                return;
            }
            
            var lines = TextUtils.BreakTextIntoLines(text);
            if (lines.Length == 0)
            {
                Debug.LogWarning("[CharacterPlayer] No lines after text processing");
                return;
            }
            
            var request = new SpeechRequest 
            { 
                TextLines = new List<string>(lines),
                ExpressionIndex = expressionIndex 
            };
            
            _speechQueue.Enqueue(request);
            
            Debug.Log($"[CharacterPlayer] Queued speech: {lines.Length} lines from text: {text.Substring(0, Math.Min(50, text.Length))}...");
            
            // Start processing if not already speaking
            if (_state != PlaybackState.Speaking && _speechCoroutine == null)
            {
                ProcessNextSpeech();
            }
        }

        /// <summary>
        /// Stop all playback and clear queues
        /// </summary>
        public void Stop()
        {
            StopIdleAnimation();
            
            if (_speechCoroutine != null)
            {
                StopCoroutine(_speechCoroutine);
                _speechCoroutine = null;
            }
            
            if (_audioSource != null && _audioSource.isPlaying)
            {
                _audioSource.Stop();
            }
            
            _speechQueue.Clear();
            
            if (_character != null && _character.IsDataLoaded)
            {
                _state = PlaybackState.Idle;
            }
            else
            {
                _state = PlaybackState.Uninitialized;
            }
        }

        /// <summary>
        /// Pause playback (both idle and speech)
        /// </summary>
        public void Pause()
        {
            if (_state == PlaybackState.Speaking && _audioSource != null)
            {
                _audioSource.Pause();
            }
            _state = PlaybackState.Paused;
        }

        /// <summary>
        /// Resume playback
        /// </summary>
        public void Resume()
        {
            if (_state == PlaybackState.Paused)
            {
                if (_audioSource != null && _audioSource.clip != null)
                {
                    _audioSource.UnPause();
                    _state = PlaybackState.Speaking;
                }
                else
                {
                    _state = PlaybackState.Idle;
                    StartIdleAnimation();
                }
            }
        }

        /// <summary>
        /// Clear all queued speech
        /// </summary>
        public void ClearQueue()
        {
            _speechQueue.Clear();
        }

        private IEnumerator LoadCharacterCoroutine(Character character)
        {
            Debug.Log($"[CharacterPlayer] Loading character: {character.Name}");
            yield return character.LoadData();
            
            Debug.Log($"[CharacterPlayer] Character loaded successfully: {character.Name}");
            OnCharacterLoadedInternal();
        }

        private void OnCharacterLoadedInternal()
        {
            // Extract idle frames (expression 0)
            LoadIdleFrames();
            
            _state = PlaybackState.Idle;
            OnCharacterLoaded?.Invoke();
            
            // Start idle animation
            if (autoPlayIdle)
            {
                StartIdleAnimation();
            }
        }

        private void LoadIdleFrames()
        {
            _idleFrames = new List<Texture2D>();
            
            if (_character == null || string.IsNullOrEmpty(_character.CharacterFolder))
            {
                Debug.LogWarning($"[CharacterPlayer] Cannot load idle frames: character or folder is null");
                return;
            }
            
            // Expression 0 folder path
            string expression0Folder = Path.Combine(_character.CharacterFolder, "drivingFrames", "expression-0");
            
            if (!Directory.Exists(expression0Folder))
            {
                Debug.LogWarning($"[CharacterPlayer] Expression 0 folder not found: {expression0Folder}");
                return;
            }
            
            // Load all PNG frames from the expression folder
            var framePaths = Directory.GetFiles(expression0Folder, "*.png")
                .OrderBy(p => p)
                .ToArray();
            
            if (framePaths.Length == 0)
            {
                Debug.LogWarning($"[CharacterPlayer] No frames found in: {expression0Folder}");
                return;
            }
            
            // Load textures from disk
            foreach (var framePath in framePaths)
            {
                byte[] fileData = File.ReadAllBytes(framePath);
                Texture2D texture = new Texture2D(2, 2);
                if (texture.LoadImage(fileData))
                {
                    _idleFrames.Add(texture);
                }
            }
            
            if (_idleFrames.Count == 0)
            {
                Debug.LogWarning($"[CharacterPlayer] No frames loaded for character: {_character.Name}");
            }
            else
            {
                Debug.Log($"[CharacterPlayer] Loaded {_idleFrames.Count} idle frames from {expression0Folder}");
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
                Texture2D currentFrame = _idleFrames[_idleFrameIndex];
                if (displayImage != null)
                {
                    displayImage.texture = currentFrame;
                }
                OnFrameUpdate?.Invoke(currentFrame);
                
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

        private void ProcessNextSpeech()
        {
            if (_speechQueue.Count == 0)
                return;
            
            _state = PlaybackState.Speaking;
            // DON'T stop idle animation yet - let it play while speech is being generated
            // AnimationPlayerLoop will stop it when first segment is ready to play
            OnSpeechStarted?.Invoke();
            
            // Start both processor and player loops for pipelining
            if (!_isSpeechProcessorRunning)
            {
                _speechCoroutine = StartCoroutine(SpeechProcessorLoop());
            }
            
            if (!_isAnimationPlayerRunning)
            {
                StartCoroutine(AnimationPlayerLoop());
            }
        }

        /// <summary>
        /// Processes queued speech requests, generating audio/animation for each line.
        /// Audio generation is serialized, but animation collection happens in parallel.
        /// Runs in parallel with AnimationPlayerLoop for pipelining.
        /// </summary>
        private IEnumerator SpeechProcessorLoop()
        {
            _isSpeechProcessorRunning = true;
            
            while (_speechQueue.Count > 0)
            {
                var request = _speechQueue.Dequeue();
                Debug.Log($"[CharacterPlayer] Processing speech request: {request.TextLines.Count} lines");
                
                foreach (var line in request.TextLines)
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;
                    
                    Debug.Log($"[CharacterPlayer] Generating line: {line.Substring(0, Math.Min(50, line.Length))}...");
                    
                    // Create pending item for this line
                    var pendingItem = new PendingSpeechItem
                    {
                        Frames = new List<Texture2D>(),
                        AudioClip = null,
                        AudioReady = false,
                        AnimationReady = false,
                        FrameStream = null
                    };
                    
                    _pendingAnimations.Enqueue(pendingItem);
                    
                    // Generate speech audio + animation
                    FrameStream frameStream = null;
                    AudioClip audioClip = null;
                    bool hasError = false;
                    
                    IEnumerator speechCoroutine = _character.SpeakAsync(
                        line,
                        request.ExpressionIndex,
                        onAudioReady: (stream, clip) =>
                        {
                            frameStream = stream;
                            audioClip = clip;
                            pendingItem.AudioClip = clip;
                            pendingItem.FrameStream = stream;
                            pendingItem.AudioReady = true;
                            Debug.Log($"[CharacterPlayer] Audio ready: {clip.length}s - can start next audio generation!");
                        },
                        onAnimationComplete: (stream) =>
                        {
                            Debug.Log($"[CharacterPlayer] Animation generation complete: {stream?.TotalExpectedFrames ?? 0} frames");
                        },
                        onError: (ex) =>
                        {
                            Debug.LogError($"[CharacterPlayer] Speech error: {ex.Message}");
                            OnError?.Invoke(ex);
                            hasError = true;
                        }
                    );
                    
                    // Start the speech generation
                    yield return speechCoroutine;
                    
                    if (hasError || audioClip == null || frameStream == null)
                    {
                        Debug.LogWarning($"[CharacterPlayer] Failed to generate speech for line");
                        pendingItem.AudioReady = true;
                        pendingItem.AnimationReady = true; // Mark as ready (but empty) so player can skip
                        continue;
                    }
                    
                    // Audio is ready! Start a separate coroutine to collect frames in parallel
                    StartCoroutine(CollectAnimationFrames(pendingItem, frameStream));
                    
                    // DON'T wait for animation - immediately continue to next line's audio generation
                    Debug.Log($"[CharacterPlayer] Audio done for line - starting next audio generation immediately!");
                }
            }
            
            _isSpeechProcessorRunning = false;
            Debug.Log("[CharacterPlayer] Speech processor loop ended");
        }
        
        /// <summary>
        /// Collects animation frames in parallel with audio generation for next segment.
        /// </summary>
        private IEnumerator CollectAnimationFrames(PendingSpeechItem item, FrameStream frameStream)
        {
            Debug.Log($"[CharacterPlayer] Starting animation frame collection in parallel...");
            
            // Collect all frames
            while (frameStream.HasMoreFrames)
            {
                var awaiter = frameStream.WaitForNext();
                yield return awaiter;
                
                if (awaiter.Texture != null)
                {
                    item.Frames.Add(awaiter.Texture);
                }
            }
            
            Debug.Log($"[CharacterPlayer] Animation frames collected: {item.Frames.Count} frames");
            
            // Mark animation as ready
            item.AnimationReady = true;
        }

        /// <summary>
        /// Plays generated animation frames synchronized with audio.
        /// Runs in parallel with SpeechProcessorLoop for pipelining.
        /// Returns to idle animation only if waiting for next segment.
        /// </summary>
        private IEnumerator AnimationPlayerLoop()
        {
            _isAnimationPlayerRunning = true;
            bool isFirstSegment = true;
            
            while (_isSpeechProcessorRunning || _pendingAnimations.Count > 0)
            {
                // Check if we need to wait for next segment
                bool needsToWait = _pendingAnimations.Count == 0 || !_pendingAnimations.Peek().IsReady;
                
                if (needsToWait && !isFirstSegment)
                {
                    // Next segment not ready - return to idle while waiting
                    Debug.Log("[CharacterPlayer] Next segment not ready - returning to idle while waiting");
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
                
                // Skip empty items
                if (item.Frames.Count == 0 || item.AudioClip == null)
                {
                    Debug.LogWarning("[CharacterPlayer] Skipping empty speech item");
                    continue;
                }
                
                // Stop idle and transition to last idle frame before playing
                if (needsToWait || isFirstSegment)
                {
                    // We were in idle (either first time or returned to idle) - do smooth transition
                    Debug.Log("[CharacterPlayer] Segment ready - transitioning from idle to speech");
                    StopIdleAnimation();
                    
                    // Transition idle to its last frame for smooth start
                    if (_idleFrames != null && _idleFrames.Count > 0)
                    {
                        Texture2D lastIdleFrame = _idleFrames[_idleFrames.Count - 1];
                        if (displayImage != null)
                        {
                            displayImage.texture = lastIdleFrame;
                        }
                        OnFrameUpdate?.Invoke(lastIdleFrame);
                        
                        // Brief pause to show transition
                        yield return new WaitForSeconds(0.04f);
                    }
                }
                else
                {
                    // Next segment was already ready - play immediately (no idle transition)
                    Debug.Log("[CharacterPlayer] Next segment ready - playing immediately");
                }
                
                Debug.Log($"[CharacterPlayer] Playing segment: {item.Frames.Count} frames, {item.AudioClip.length}s");
                
                // Play this segment with its audio
                yield return PlayFramesSynchronized(item.Frames, item.AudioClip);
                
                // After playing, we're no longer in first segment
                isFirstSegment = false;
            }
            
            _isAnimationPlayerRunning = false;
            
            Debug.Log("[CharacterPlayer] Animation player loop ended");
            
            // Speech complete
            _speechCoroutine = null;
            _state = PlaybackState.Idle;
            OnSpeechEnded?.Invoke();
            
            // Return to idle (starting from frame 0 for smooth loop)
            _idleFrameIndex = 0;
            _idleForward = true;
            
            if (autoPlayIdle)
            {
                StartIdleAnimation();
            }
        }

        private List<Texture2D> GetTransitionToIdleFrames()
        {
            List<Texture2D> transition = new List<Texture2D>();
            
            if (_idleFrames != null && _idleFrames.Count > 0)
            {
                // Add last few idle frames in reverse to smoothly return to frame 0
                int transitionFrameCount = Math.Min(3, _idleFrames.Count);
                for (int i = transitionFrameCount - 1; i >= 0; i--)
                {
                    transition.Add(_idleFrames[i]);
                }
            }
            
            return transition;
        }

        private IEnumerator PlayFramesSynchronized(List<Texture2D> frames, AudioClip audioClip)
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
            
            // Play frames at calculated rate
            for (int i = 0; i < frames.Count && _state == PlaybackState.Speaking; i++)
            {
                if (displayImage != null)
                {
                    displayImage.texture = frames[i];
                }
                OnFrameUpdate?.Invoke(frames[i]);
                
                yield return new WaitForSeconds(frameInterval);
            }
            
            // Wait for audio to finish
            while (_audioSource.isPlaying && _state == PlaybackState.Speaking)
            {
                yield return null;
            }
        }

        private void OnDestroy()
        {
            Stop();
        }
    }
}

