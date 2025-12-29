using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

namespace LiveTalk.Utils
{
    public class AudioQueuePlayer : MonoBehaviour
    {
        private readonly Queue<AudioClip> _clipQueue = new();
        private AudioSource _audioSource;
        private int _totalExpectedClips;
        private int _clipsReceived;
        private bool _isPlaying;
        private bool _loadingComplete;
        
        public event Action<bool> OnPlaybackStatusChanged; // true when started, false when finished
        public event Action OnQueueEmpty;
        public event Action<float> OnProgressUpdated; // 0-1 representing playback progress

        public bool IsPlaying => _isPlaying;
        public bool IsLoadingComplete => _loadingComplete;
        public int QueueCount => _clipQueue.Count;
        public int TotalExpectedClips => _totalExpectedClips;
        public int ClipsReceived => _clipsReceived;

        private void Awake()
        {
            _audioSource = GetComponent<AudioSource>();
            if (_audioSource == null)
            {
                _audioSource = gameObject.AddComponent<AudioSource>();
            }
            
            _audioSource.playOnAwake = false;
            _audioSource.loop = false;
        }

        public void Initialize(int expectedClipCount)
        {
            StopAllCoroutines();
            _clipQueue.Clear();
            _totalExpectedClips = expectedClipCount;
            _clipsReceived = 0;
            _isPlaying = false;
            _loadingComplete = false;
            
            Debug.Log($"AudioQueuePlayer initialized with {expectedClipCount} expected clips");
        }

        public void SetLoadingComplete()
        {
            _loadingComplete = true;
            Debug.Log("AudioQueuePlayer loading marked as complete");
            
            // If there are no clips in the queue and we're not playing, notify that the queue is empty
            if (!_isPlaying && _clipQueue.Count == 0)
            {
                OnQueueEmpty?.Invoke();
            }
        }

        public void EnqueueClip(AudioClip clip)
        {
            if (clip == null)
                return;
                
            _clipQueue.Enqueue(clip);
            _clipsReceived++;
            
            Debug.Log($"Clip enqueued. Queue size: {_clipQueue.Count}, Received: {_clipsReceived}/{_totalExpectedClips}");
            
            // If we're not currently playing, start playback
            if (!_isPlaying)
            {
                StartCoroutine(PlayQueueCoroutine());
            }
        }

        public void Stop()
        {
            StopAllCoroutines();
            _audioSource.Stop();
            _isPlaying = false;
            _clipQueue.Clear();
            OnPlaybackStatusChanged?.Invoke(false);
            Debug.Log("AudioQueuePlayer playback stopped");
        }

        public void Pause()
        {
            if (_isPlaying)
            {
                _audioSource.Pause();
            }
        }

        public void Resume()
        {
            if (_isPlaying)
            {
                _audioSource.UnPause();
            }
        }

        private IEnumerator PlayQueueCoroutine()
        {
            _isPlaying = true;
            OnPlaybackStatusChanged?.Invoke(true);
            Debug.Log("Starting audio queue playback");
            
            while (_clipQueue.Count > 0 || !_loadingComplete)
            {
                // If there's a clip in the queue, play it
                if (_clipQueue.Count > 0)
                {
                    AudioClip nextClip = _clipQueue.Dequeue();
                    _audioSource.clip = nextClip;
                    _audioSource.Play();
                    
                    float startTime = Time.time;
                    
                    // Wait until the clip is done playing
                    while (_audioSource.isPlaying)
                    {
                        float progress = (_clipsReceived > 0) ? 
                            (_clipsReceived - _clipQueue.Count - 1) / (float)_totalExpectedClips : 0;
                        
                        // Add progress within the current clip
                        if (_audioSource.clip != null && _audioSource.clip.length > 0)
                        {
                            float clipProgress = _audioSource.time / _audioSource.clip.length;
                            progress += clipProgress / _totalExpectedClips;
                        }
                        
                        OnProgressUpdated?.Invoke(progress);
                        yield return null;
                    }
                    
                    Debug.Log($"Clip finished playing after {Time.time - startTime} seconds. Remaining in queue: {_clipQueue.Count}");
                }
                else if (!_loadingComplete)
                {
                    // Wait for more clips to be added
                    Debug.Log("Waiting for more clips to be added to the queue...");
                    yield return new WaitForSeconds(0.1f);
                }
                else
                {
                    // No more clips and loading is complete
                    break;
                }
            }
            
            _isPlaying = false;
            OnPlaybackStatusChanged?.Invoke(false);
            OnQueueEmpty?.Invoke();
            Debug.Log("Audio queue playback complete");
        }

        /// <summary>
        /// Plays a single audio clip immediately, handling initialization and completion automatically
        /// </summary>
        /// <param name="clip">The clip to play</param>
        public void PlayClipImmediate(AudioClip clip)
        {
            if (clip == null)
                return;
                
            // Clear any existing playback
            Stop();
            
            // Initialize for a single clip
            Initialize(1);
            
            // Add the clip
            EnqueueClip(clip);
            
            // Mark as complete
            SetLoadingComplete();
            
            Debug.Log($"Playing clip immediately: {clip.name}");
        }
        
        /// <summary>
        /// Plays multiple audio clips in sequence, handling initialization and completion automatically
        /// </summary>
        /// <param name="clips">The clips to play in sequence</param>
        public void PlayClipsImmediate(IEnumerable<AudioClip> clips)
        {
            if (clips == null)
                return;
                
            var clipsList = clips.Where(c => c != null).ToList();
            if (clipsList.Count == 0)
                return;
                
            // Clear any existing playback
            Stop();
            
            // Initialize for all clips
            Initialize(clipsList.Count);
            
            // Add all clips
            foreach (var clip in clipsList)
            {
                EnqueueClip(clip);
            }
            
            // Mark as complete
            SetLoadingComplete();
            
            Debug.Log($"Playing {clipsList.Count} clips immediately");
        }
    }
}

