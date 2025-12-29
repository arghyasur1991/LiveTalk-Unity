using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace LiveTalk.API
{
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
            
            // Break text into lines using TextUtils (same as SpeechPlaybackManager)
            var lines = LiveTalk.Utils.TextUtils.BreakTextIntoLines(text);
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
            
            while (_state == PlaybackState.Idle)
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
            
            var request = _speechQueue.Dequeue();
            _speechCoroutine = StartCoroutine(ProcessSpeechWithTransition(request.TextLines, request.ExpressionIndex));
        }

        private IEnumerator ProcessSpeechWithTransition(List<string> textLines, int expressionIndex)
        {
            _state = PlaybackState.Speaking;
            StopIdleAnimation();
            OnSpeechStarted?.Invoke();
            
            Debug.Log($"[CharacterPlayer] Starting speech: {textLines.Count} lines");
            
            List<Texture2D> allSpeechFrames = new List<Texture2D>();
            List<AudioClip> allAudioClips = new List<AudioClip>();
            
            // Process each line separately (like SpeechPlaybackManager)
            foreach (var line in textLines)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                
                Debug.Log($"[CharacterPlayer] Processing line: {line.Substring(0, Math.Min(50, line.Length))}...");
                
                // Generate speech audio + animation for this line
                FrameStream frameStream = null;
                AudioClip audioClip = null;
                bool audioReady = false;
                bool hasError = false;
                
                IEnumerator speechCoroutine = _character.SpeakAsync(
                    line,
                    expressionIndex,
                    onAudioReady: (stream, clip) =>
                    {
                        frameStream = stream;
                        audioClip = clip;
                        audioReady = true;
                        Debug.Log($"[CharacterPlayer] Audio ready for line");
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
                
                yield return speechCoroutine;
                
                if (hasError)
                {
                    continue; // Skip this line but continue with others
                }
                
                // Wait for audio to be ready
                yield return new WaitUntil(() => audioReady);
                
                if (audioClip == null || frameStream == null)
                {
                    Debug.LogError("[CharacterPlayer] Audio or frame stream is null for line");
                    continue;
                }
                
                // Collect all frames for this line
                List<Texture2D> lineFrames = new List<Texture2D>();
                while (frameStream.HasMoreFrames)
                {
                    var awaiter = frameStream.WaitForNext();
                    yield return awaiter;
                    if (awaiter.Texture != null)
                    {
                        lineFrames.Add(awaiter.Texture);
                    }
                }
                
                Debug.Log($"[CharacterPlayer] Collected {lineFrames.Count} frames for line");
                
                // Add to combined lists
                allSpeechFrames.AddRange(lineFrames);
                allAudioClips.Add(audioClip);
            }
            
            if (allAudioClips.Count == 0)
            {
                Debug.LogError("[CharacterPlayer] No audio generated for any lines");
                _state = PlaybackState.Idle;
                _speechCoroutine = null;
                StartIdleAnimation();
                ProcessNextSpeech();
                yield break;
            }
            
            Debug.Log($"[CharacterPlayer] Total collected: {allSpeechFrames.Count} frames, {allAudioClips.Count} audio clips");
            
            // Concatenate audio clips
            AudioClip combinedAudio = allAudioClips.Count == 1 
                ? allAudioClips[0] 
                : LiveTalk.Utils.AudioUtils.ConcatenateAudioClips(allAudioClips, 16000);
            
            // Append transition frames to return to idle frame 0
            var transitionFrames = GetTransitionToIdleFrames();
            allSpeechFrames.AddRange(transitionFrames);
            
            // Play frames synchronized with combined audio
            yield return PlayFramesSynchronized(allSpeechFrames, combinedAudio);
            
            Debug.Log("[CharacterPlayer] Speech playback complete");
            
            // Transition back to idle
            _state = PlaybackState.Idle;
            _idleFrameIndex = 0; // Start from frame 0
            _idleForward = true;
            _speechCoroutine = null;
            OnSpeechEnded?.Invoke();
            
            // Start idle animation
            StartIdleAnimation();
            
            // Process next speech in queue
            if (_speechQueue.Count > 0)
            {
                ProcessNextSpeech();
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

