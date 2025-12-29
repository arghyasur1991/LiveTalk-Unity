using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace LiveTalk.API
{
    /// <summary>
    /// Orchestrates multi-character turn-based dialogue using CharacterPlayer instances.
    /// Handles speaker switching, audio coordination, and visual display management.
    /// 
    /// Usage:
    /// 1. Create CharacterPlayer instances for each character
    /// 2. Register them with RegisterCharacter()
    /// 3. Queue dialogue with QueueDialogue()
    /// 4. Orchestrator handles turn-based speaking automatically
    /// </summary>
    public class DialogueOrchestrator : MonoBehaviour
    {
        [Header("Display Settings")]
        [SerializeField] private RawImage displayTarget;
        [SerializeField] private bool autoHideInactiveSpeakers = true;
        
        // Registered characters
        private Dictionary<string, CharacterPlayer> _characterPlayers = new Dictionary<string, CharacterPlayer>();
        
        // Dialogue queue
        private Queue<DialogueSegment> _dialogueQueue = new Queue<DialogueSegment>();
        private bool _isProcessing = false;
        private Coroutine _dialogueCoroutine;
        
        // Current state
        private CharacterPlayer _currentSpeaker;
        
        // Events
        public event Action<string> OnSpeakerChanged; // character ID
        public event Action<Texture2D> OnFrameUpdate;
        public event Action OnDialogueStarted;
        public event Action OnDialogueEnded;
        public event Action<Exception> OnError;
        
        // Public properties
        public bool IsPlaying => _isProcessing;
        public int QueuedDialogueCount => _dialogueQueue.Count;
        public string CurrentSpeakerId { get; private set; }
        
        /// <summary>
        /// Dialogue segment for multi-character conversations
        /// </summary>
        public class DialogueSegment
        {
            public string CharacterId { get; set; }
            public string Text { get; set; }
            public int ExpressionIndex { get; set; }
            public bool WithAnimation { get; set; } = true;
        }
        
        /// <summary>
        /// Register a character for dialogue orchestration
        /// </summary>
        /// <param name="characterId">Unique ID for this character (e.g., "detective", "narrator")</param>
        /// <param name="player">CharacterPlayer instance for this character</param>
        /// <param name="visualElement">Optional: UI element to show/hide for this character</param>
        public void RegisterCharacter(string characterId, CharacterPlayer player, GameObject visualElement = null)
        {
            if (_characterPlayers.ContainsKey(characterId))
            {
                Debug.LogWarning($"[DialogueOrchestrator] Character {characterId} already registered, replacing");
                _characterPlayers[characterId] = player;
            }
            else
            {
                _characterPlayers.Add(characterId, player);
            }
            
            // Subscribe to player events
            player.OnFrameUpdate += OnCharacterFrameUpdate;
            player.OnSpeechStarted += () => OnCharacterSpeechStarted(characterId);
            player.OnSpeechEnded += () => OnCharacterSpeechEnded(characterId);
            player.OnError += OnError;
            
            Debug.Log($"[DialogueOrchestrator] Registered character: {characterId}");
        }
        
        /// <summary>
        /// Unregister a character
        /// </summary>
        public void UnregisterCharacter(string characterId)
        {
            if (_characterPlayers.TryGetValue(characterId, out var player))
            {
                player.OnFrameUpdate -= OnCharacterFrameUpdate;
                player.OnError -= OnError;
                _characterPlayers.Remove(characterId);
                Debug.Log($"[DialogueOrchestrator] Unregistered character: {characterId}");
            }
        }
        
        /// <summary>
        /// Queue a single dialogue line
        /// </summary>
        /// <param name="withAnimation">If false, plays audio only (useful for characters without avatars like narrator)</param>
        public void QueueDialogue(string characterId, string text, int expressionIndex = 0, bool withAnimation = true)
        {
            if (!_characterPlayers.ContainsKey(characterId))
            {
                Debug.LogError($"[DialogueOrchestrator] Character {characterId} not registered!");
                return;
            }
            
            _dialogueQueue.Enqueue(new DialogueSegment
            {
                CharacterId = characterId,
                Text = text,
                ExpressionIndex = expressionIndex,
                WithAnimation = withAnimation
            });
            
            Debug.Log($"[DialogueOrchestrator] Queued dialogue for {characterId}: {text.Substring(0, Math.Min(50, text.Length))}... (Queue size: {_dialogueQueue.Count}, Animation: {withAnimation})");
            
            // Start processing if not already running
            if (!_isProcessing)
            {
                StartDialogue();
            }
        }
        
        /// <summary>
        /// Queue multiple dialogue segments at once
        /// </summary>
        public void QueueDialogueBatch(List<DialogueSegment> segments)
        {
            foreach (var segment in segments)
            {
                if (!_characterPlayers.ContainsKey(segment.CharacterId))
                {
                    Debug.LogError($"[DialogueOrchestrator] Character {segment.CharacterId} not registered, skipping segment!");
                    continue;
                }
                
                _dialogueQueue.Enqueue(segment);
            }
            
            Debug.Log($"[DialogueOrchestrator] Queued {segments.Count} dialogue segments (Queue size: {_dialogueQueue.Count})");
            
            // Start processing if not already running
            if (!_isProcessing)
            {
                StartDialogue();
            }
        }
        
        /// <summary>
        /// Start dialogue processing
        /// </summary>
        private void StartDialogue()
        {
            if (_dialogueCoroutine != null)
            {
                StopCoroutine(_dialogueCoroutine);
            }
            
            _dialogueCoroutine = StartCoroutine(ProcessDialogueQueue());
        }
        
        /// <summary>
        /// Stop all dialogue and clear queue
        /// </summary>
        public void Stop()
        {
            if (_dialogueCoroutine != null)
            {
                StopCoroutine(_dialogueCoroutine);
                _dialogueCoroutine = null;
            }
            
            // Stop all character players
            foreach (var player in _characterPlayers.Values)
            {
                player.Stop();
            }
            
            _dialogueQueue.Clear();
            _isProcessing = false;
            _currentSpeaker = null;
            CurrentSpeakerId = null;
            
            OnDialogueEnded?.Invoke();
            
            Debug.Log("[DialogueOrchestrator] Stopped");
        }
        
        /// <summary>
        /// Clear the dialogue queue (but don't stop current speech)
        /// </summary>
        public void ClearQueue()
        {
            _dialogueQueue.Clear();
            Debug.Log("[DialogueOrchestrator] Queue cleared");
        }
        
        /// <summary>
        /// Main dialogue processing loop
        /// </summary>
        private IEnumerator ProcessDialogueQueue()
        {
            _isProcessing = true;
            OnDialogueStarted?.Invoke();
            
            Debug.Log("[DialogueOrchestrator] Started processing dialogue queue");
            
            while (_dialogueQueue.Count > 0)
            {
                var segment = _dialogueQueue.Dequeue();
                
                if (!_characterPlayers.TryGetValue(segment.CharacterId, out var player))
                {
                    Debug.LogWarning($"[DialogueOrchestrator] Character {segment.CharacterId} not found, skipping");
                    continue;
                }
                
                // Switch to this speaker
                SwitchSpeaker(segment.CharacterId, player);
                
                // Queue speech to the character's player
                player.QueueSpeech(segment.Text, segment.ExpressionIndex, segment.WithAnimation);
                
                Debug.Log($"[DialogueOrchestrator] {segment.CharacterId} speaking: {segment.Text.Substring(0, Math.Min(50, segment.Text.Length))}... (Animation: {segment.WithAnimation})");
                
                // Wait for this character to finish speaking
                while (player.IsPlaying || player.QueuedSpeechCount > 0)
                {
                    yield return new WaitForSeconds(0.1f);
                }
                
                Debug.Log($"[DialogueOrchestrator] {segment.CharacterId} finished");
            }
            
            _isProcessing = false;
            _currentSpeaker = null;
            CurrentSpeakerId = null;
            
            OnDialogueEnded?.Invoke();
            
            Debug.Log("[DialogueOrchestrator] Dialogue queue processing complete");
        }
        
        /// <summary>
        /// Switch active speaker
        /// </summary>
        private void SwitchSpeaker(string characterId, CharacterPlayer player)
        {
            // Stop previous speaker if different
            if (_currentSpeaker != null && _currentSpeaker != player)
            {
                _currentSpeaker.Stop();
            }
            
            _currentSpeaker = player;
            CurrentSpeakerId = characterId;
            
            OnSpeakerChanged?.Invoke(characterId);
            
            Debug.Log($"[DialogueOrchestrator] Switched speaker to: {characterId}");
        }
        
        /// <summary>
        /// Forward frame updates from current speaker
        /// </summary>
        private void OnCharacterFrameUpdate(Texture2D frame)
        {
            if (displayTarget != null)
            {
                displayTarget.texture = frame;
            }
            
            OnFrameUpdate?.Invoke(frame);
        }
        
        private void OnCharacterSpeechStarted(string characterId)
        {
            Debug.Log($"[DialogueOrchestrator] {characterId} started speaking");
        }
        
        private void OnCharacterSpeechEnded(string characterId)
        {
            Debug.Log($"[DialogueOrchestrator] {characterId} ended speaking");
        }
        
        private void OnDestroy()
        {
            // Unsubscribe from all players
            foreach (var player in _characterPlayers.Values)
            {
                if (player != null)
                {
                    player.OnFrameUpdate -= OnCharacterFrameUpdate;
                    player.OnError -= OnError;
                }
            }
            
            _characterPlayers.Clear();
        }
    }
}

