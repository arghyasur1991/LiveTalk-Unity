using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using LiveTalk.Utils;

namespace LiveTalk.API
{
    /// <summary>
    /// Manages generic speech generation and playback for multiple characters.
    /// Handles pipelined audio/animation processing and synchronized playback.
    /// This is a reusable component - game-specific logic should be in higher-level managers.
    /// </summary>
    public class SpeechPlaybackManager : MonoBehaviour
    {
        private AudioQueuePlayer _audioQueuePlayer;
        
        // Animation support
        private const float ANIMATION_FPS = 25f;
        private const float FRAME_INTERVAL = 1f / ANIMATION_FPS;
        private Coroutine _animationCoroutine;
        private int _currentFrameIndex;
        private float _lastFrameTime;
        
        // Pending animation data for pipelined playback
        private Queue<(List<Texture2D> frames, float audioDuration)> _pendingAnimations = new();
        private bool _isAnimationPlayerRunning;
        
        // Queue for pending speech items (to maintain order while allowing pipelined generation)
        // frameStream can be null for audio-only items (like silence)
        private Queue<(FrameStream frameStream, AudioClip audioClip)> _pendingSpeechItems = new();
        private bool _isSpeechProcessorRunning;
        
        // Events for animation
        public event Action<Texture2D> OnFrameUpdate;  // Called when a new frame should be displayed
        public event Action OnAnimationStarted;
        public event Action OnAnimationEnded;

        public bool IsPlaying => _audioQueuePlayer != null && _audioQueuePlayer.IsPlaying;
        public AudioQueuePlayer AudioPlayer => _audioQueuePlayer;

        private void Awake()
        {
            // Initialize audio queue player
            _audioQueuePlayer = GetComponent<AudioQueuePlayer>();
            if (_audioQueuePlayer == null)
            {
                _audioQueuePlayer = gameObject.AddComponent<AudioQueuePlayer>();
            }
        }

        /// <summary>
        /// Initialize the playback manager for a new speech sequence
        /// </summary>
        /// <param name="expectedSegmentCount">Number of audio segments expected</param>
        public void Initialize(int expectedSegmentCount)
        {
            _audioQueuePlayer.Initialize(expectedSegmentCount);
            
            // Clear pending queues
            _pendingAnimations.Clear();
            _pendingSpeechItems.Clear();
            
            // Start the processor coroutines if not running
            if (!_isAnimationPlayerRunning)
            {
                StartCoroutine(AnimationPlayerLoop());
            }
            if (!_isSpeechProcessorRunning)
            {
                StartCoroutine(SpeechProcessorLoop());
            }
            
            Debug.Log($"[SpeechPlaybackManager] Initialized with {expectedSegmentCount} expected segments");
        }

        /// <summary>
        /// Mark loading as complete - playback will finish after all queued items play
        /// </summary>
        public void SetLoadingComplete()
        {
            _audioQueuePlayer.SetLoadingComplete();
            Debug.Log("[SpeechPlaybackManager] Loading marked as complete");
        }

        /// <summary>
        /// Queue a single speech segment with optional animation
        /// </summary>
        /// <param name="character">Character speaking</param>
        /// <param name="text">Text to speak</param>
        /// <param name="withAnimation">Whether to generate animation</param>
        /// <param name="breakIntoLines">Whether to automatically break text into sentences</param>
        public async Task QueueSpeechAsync(
            Character character, 
            string text, 
            bool withAnimation = true,
            bool breakIntoLines = true)
        {
            if (string.IsNullOrEmpty(text))
                return;

            if (breakIntoLines)
            {
                var lines = TextUtils.BreakTextIntoLines(text);
                foreach (var line in lines)
                {
                    await QueueSingleSpeechAsync(character, line, withAnimation);
                }
            }
            else
            {
                await QueueSingleSpeechAsync(character, text, withAnimation);
            }
        }

        /// <summary>
        /// Queue multiple pre-processed speech segments
        /// </summary>
        /// <param name="segments">List of (character, text, withAnimation) tuples</param>
        public async Task QueueSpeechSegmentsAsync(
            List<(Character character, string text, bool withAnimation)> segments)
        {
            foreach (var (character, text, withAnimation) in segments)
            {
                await QueueSingleSpeechAsync(character, text, withAnimation);
            }
        }

        /// <summary>
        /// Queue raw audio clip (for silence or pre-generated audio)
        /// </summary>
        /// <param name="clip">Audio clip to queue</param>
        public void QueueAudioClip(AudioClip clip)
        {
            if (clip != null)
            {
                _pendingSpeechItems.Enqueue((null, clip));  // null frameStream = audio-only
            }
        }

        /// <summary>
        /// Add text to segments list, breaking it into lines
        /// </summary>
        /// <param name="text">Text to add</param>
        /// <param name="character">Character speaking</param>
        /// <param name="segments">List to add segments to</param>
        public void AddTextToSegments(
            string text, 
            Character character,
            List<(Character character, string text)> segments)
        {
            var lines = TextUtils.BreakTextIntoLines(text);
            foreach (string line in lines)
            {
                segments.Add((character, line));
            }
        }

        /// <summary>
        /// Stop playback and clear all queues
        /// </summary>
        public void Stop()
        {
            _audioQueuePlayer?.Stop();
            _pendingAnimations.Clear();
            _pendingSpeechItems.Clear();
            StopAnimation();
        }

        /// <summary>
        /// Pause audio playback
        /// </summary>
        public void Pause()
        {
            _audioQueuePlayer?.Pause();
        }

        /// <summary>
        /// Resume audio playback
        /// </summary>
        public void Resume()
        {
            _audioQueuePlayer?.Resume();
        }

        /// <summary>
        /// Stop any playing animation
        /// </summary>
        public void StopAnimation()
        {
            if (_animationCoroutine != null)
            {
                StopCoroutine(_animationCoroutine);
                _animationCoroutine = null;
            }
            OnAnimationEnded?.Invoke();
        }

        // ==================== INTERNAL METHODS ====================

        /// <summary>
        /// Queue a single speech segment (internal, no line breaking)
        /// </summary>
        private async Task QueueSingleSpeechAsync(Character character, string text, bool withAnimation)
        {
            if (!character.IsDataLoaded || !withAnimation)
            {
                // Audio only - this shouldn't happen as Character.SpeakAsync requires loaded character
                Debug.LogWarning($"[SpeechPlaybackManager] Character {character.Name} not loaded or no animation requested");
                return;
            }

            try
            {
                var audioTcs = new TaskCompletionSource<(FrameStream, AudioClip)>();
                
                // Use LiveTalk Character's SpeakAsync coroutine
                IEnumerator speakCoroutine = character.SpeakAsync(
                    text,
                    expressionIndex: withAnimation ? 0 : -1,
                    onAudioReady: (frameStream, audioClip) =>
                    {
                        // Audio is ready - signal to continue with next segment's generation
                        audioTcs.SetResult((frameStream, audioClip));
                    },
                    onAnimationComplete: (frameStream) =>
                    {
                        Debug.Log($"[SpeechPlaybackManager] Animation generation complete: {frameStream?.TotalExpectedFrames ?? 0} frames");
                    },
                    onError: (error) =>
                    {
                        Debug.LogError($"[SpeechPlaybackManager] Speech generation failed: {error.Message}");
                        audioTcs.SetResult((null, null));
                    }
                );
                
                // Start the coroutine
                StartCoroutine(speakCoroutine);
                
                // Wait for audio to be ready (next segment can start generating after this)
                var (frameStream, audioClip) = await audioTcs.Task;
                
                if (audioClip != null)
                {
                    // Queue for ordered processing - maintains segment order
                    // SpeechProcessorLoop will collect frames and enqueue audio in order
                    _pendingSpeechItems.Enqueue((frameStream, audioClip));
                    Debug.Log($"[SpeechPlaybackManager] Audio ready, queued for ordered playback. Queue size: {_pendingSpeechItems.Count}");
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[SpeechPlaybackManager] Failed to generate animated speech: {ex.Message}");
            }
        }

        /// <summary>
        /// Processes speech items in order: collects frames, then enqueues audio.
        /// This ensures segments play in the correct order even with pipelined generation.
        /// </summary>
        private IEnumerator SpeechProcessorLoop()
        {
            _isSpeechProcessorRunning = true;
            
            while (true)
            {
                // Wait for pending speech item
                while (_pendingSpeechItems.Count == 0)
                {
                    yield return null;
                }
                
                // Get next speech item (in order)
                var (frameStream, audioClip) = _pendingSpeechItems.Dequeue();
                
                if (frameStream != null)
                {
                    // Collect all frames first
                    var frames = new List<Texture2D>();
                    while (frameStream.HasMoreFrames)
                    {
                        var awaiter = frameStream.WaitForNext();
                        yield return awaiter;
                        
                        if (awaiter.Texture != null)
                        {
                            frames.Add(awaiter.Texture);
                        }
                    }
                    
                    Debug.Log($"[SpeechPlaybackManager] Collected {frames.Count} frames, now enqueueing audio for playback");
                    
                    // NOW enqueue audio - it will start playing via AudioQueuePlayer
                    _audioQueuePlayer.EnqueueClip(audioClip);
                    
                    // Queue the animation for synchronized playback
                    if (frames.Count > 0)
                    {
                        _pendingAnimations.Enqueue((frames, audioClip.length));
                    }
                }
                else
                {
                    // Audio only - no animation
                    _audioQueuePlayer.EnqueueClip(audioClip);
                }
            }
        }
        
        /// <summary>
        /// Continuously processes pending animations for synchronized playback
        /// </summary>
        private IEnumerator AnimationPlayerLoop()
        {
            _isAnimationPlayerRunning = true;
            
            while (true)
            {
                // Wait for pending animation
                while (_pendingAnimations.Count == 0)
                {
                    yield return null;
                }
                
                // Get next animation to play
                var (frames, audioDuration) = _pendingAnimations.Dequeue();
                
                // Play frames synchronized with audio
                yield return PlayFramesSynchronized(frames, audioDuration);
            }
        }
        
        /// <summary>
        /// Play collected frames at 25 FPS synchronized with audio
        /// </summary>
        private IEnumerator PlayFramesSynchronized(List<Texture2D> frames, float audioDuration)
        {
            if (frames.Count == 0)
            {
                yield break;
            }
            
            _currentFrameIndex = 0;
            _lastFrameTime = Time.time;
            
            OnAnimationStarted?.Invoke();
            
            // Calculate actual frame interval based on frame count and audio duration
            float actualFrameInterval = audioDuration / frames.Count;
            
            // Play frames at calculated rate
            while (_currentFrameIndex < frames.Count)
            {
                float currentTime = Time.time;
                if (currentTime - _lastFrameTime >= actualFrameInterval)
                {
                    OnFrameUpdate?.Invoke(frames[_currentFrameIndex]);
                    _currentFrameIndex++;
                    _lastFrameTime = currentTime;
                }
                yield return null;
            }
            
            OnAnimationEnded?.Invoke();
            
            Debug.Log($"[SpeechPlaybackManager] Animation playback complete: {frames.Count} frames");
        }
    }
}

