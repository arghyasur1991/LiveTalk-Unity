using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace LiveTalk.Samples
{
    using API;
    using UnityEngine.Video;

    /// <summary>
    /// Demo showcasing the integrated LivePortrait + MuseTalk workflow
    /// 
    /// Workflow:
    /// 1. User provides a single source image (portrait)
    /// 2. User provides driving frames (expressions/head movements) 
    /// 3. User provides audio for lip sync
    /// 4. LivePortrait generates animated textures from source + driving frames
    /// 5. MuseTalk applies lip sync to the animated textures
    /// 6. Result: Complete talking head animation with expressions and lip sync
    /// </summary>
    public class LiveTalkDemo : MonoBehaviour
    {
        [Header("Input Assets")]
        [SerializeField] private Texture2D sourceImage;
        [SerializeField] private AudioClip audioClip;
        public AudioSource audioSource;
        
        [Header("Character Creator Settings")]
        [SerializeField] private string characterName = "Character";
        [SerializeField] private Gender characterGender = Gender.Female;
        [SerializeField] private Pitch characterPitch = Pitch.Moderate;
        [SerializeField] private Speed characterSpeed = Speed.Moderate;
        [SerializeField] private string characterIntro = "Hello, I am a character.";
        
        [Header("Character Loading Settings")]
        [SerializeField] private string characterIdToLoad = "";
        [SerializeField] private string textToSpeak = "Hello, this is a test message.";
        
        [Header("Driving Frames Folder (Optional - overrides array)")]
        [Tooltip("Path to folder containing driving frame images (relative to StreamingAssets)")]
        [SerializeField] private string drivingFramesFolderPath = "LivePortrait/driving_frames";

        [Header("Talking Head Avatar Folder (Optional - overrides array)")]
        [Tooltip("Path to folder containing talking head images (relative to StreamingAssets)")]
        [SerializeField] private string talkingHeadFolderPath = "LivePortrait/avatar";

        [Header("Video Player (Optional - overrides array)")]
        [Tooltip("Video player to use for driving frames")]
        [SerializeField] private VideoPlayer videoPlayer;

        [Header("Output Settings")]
        [Tooltip("Save generated frames as PNG images")]
        [SerializeField] private bool saveVideoOutput = false;
        [Tooltip("Output directory for PNG frames (relative to StreamingAssets)")]
        [SerializeField] private string videoOutputPath = "LiveTalk/output_frames";

        [Header("UI")]
        [SerializeField] private Button generateButton;
        [SerializeField] private Button generateTalkingHeadButton;
        [SerializeField] private Button createCharacterButton;
        [SerializeField] private Button loadCharacterButton;
        [SerializeField] private Button speakButton;
        [SerializeField] private TMP_Text statusText;
        [SerializeField] private TMP_Text fpsText;
        [SerializeField] private RawImage previewImage;
        [SerializeField] private Slider progressSlider;
        [SerializeField] private bool generateVoiceOnly = false;
        
        private LiveTalkAPI _api;
        private bool _isPlayingAudio = false;
        private bool _overrideAudio = false;
        private Coroutine _previewCoroutine;
        
        // Frame streaming and preview management
        private readonly List<Texture2D> _frameBuffer = new();
        private bool _isGenerating = false;
        private Coroutine _frameGenerationCoroutine;
        private int _currentPreviewFrame = 0;
        private float _lastFrameTime = 0f;
        private const float PREVIEW_FPS = 25f;
        private const float FRAME_INTERVAL = 1f / PREVIEW_FPS;
        private int _totalExpectedFrames = 0;
        
        // Video output state
        private string _outputDirectory = "";
        private int _savedFrameCount = 0;
        
        // Character management
        private Character _loadedCharacter = null;

        void Awake()
        {
            if (videoPlayer != null)
            {
                videoPlayer.isLooping = false;
                videoPlayer.playOnAwake = false;
                videoPlayer.skipOnDrop = false;
                videoPlayer.Prepare();
            }
        }
        void Start()
        {
            SetupUI();
            InitializeAPI();
        }
        
        void SetupUI()
        {
            if (generateButton != null)
                generateButton.onClick.AddListener(() => StartCoroutine(GenerateAnimatedOnly()));

            if (generateTalkingHeadButton != null)
                generateTalkingHeadButton.onClick.AddListener(() => StartCoroutine(GenerateTalkingHead()));
                
            if (createCharacterButton != null)
                createCharacterButton.onClick.AddListener(() => StartCoroutine(CreateCharacter()));
                
            if (loadCharacterButton != null)
                loadCharacterButton.onClick.AddListener(() => StartCoroutine(LoadCharacter()));
                
            if (speakButton != null)
                speakButton.onClick.AddListener(() => StartCoroutine(SpeakWithLoadedCharacter()));
                
            UpdateStatus("Ready to initialize...");
        }

        bool IsMobile()
        {
            return Application.platform == RuntimePlatform.IPhonePlayer || Application.platform == RuntimePlatform.Android;
        }
        
        void InitializeAPI()
        {
            try
            {
                UpdateStatus("Initializing LiveTalk...");
                var logLevel = LogLevel.Info;
                _api = LiveTalkAPI.Instance;
                MemoryUsage memoryUsage = IsMobile() ? MemoryUsage.Optimal : MemoryUsage.Balanced;
                _api.Initialize(logLevel, memoryUsage: memoryUsage);
                
                UpdateStatus("LiveTalk and CharacterFactory initialized successfully!");
                SetButtonsEnabled(true);
            }
            catch (System.Exception e)
            {
                UpdateStatus($"Initialization error: {e.Message}");
                SetButtonsEnabled(false);
                Debug.LogError($"LiveTalkDemo initialization failed: {e}");
            }
        }
        
        /// <summary>
        /// Generate only animated textures (LivePortrait only)
        /// </summary>
        IEnumerator GenerateAnimatedOnly()
        {
            if (!ValidateInputsForAnimatedOnly()) yield break;
            
            UpdateStatus("Generating animated textures only...");
            SetButtonsEnabled(false);
            
            // Clear previous results and reset state
            ClearFrameBuffer();
            _currentPreviewFrame = 0;
            _isGenerating = true;
            _lastFrameTime = Time.time;
            _isPlayingAudio = false;
            
            // Start frame generation in background
            if (videoPlayer != null)
            {
                _frameGenerationCoroutine = StartCoroutine(GenerateFramesAsyncFromVideo());
            }
            else
            {
                _frameGenerationCoroutine = StartCoroutine(GenerateFramesAsync());
            }
            
            // Start preview animation immediately (will show frames as they become available)
            StartStreamingPreviewAnimation();
            
            // Wait for generation to complete
            while (_isGenerating)
            {
                yield return null;
            }
            
            // Final status update
            if (_frameBuffer.Count > 0)
            {
                UpdateStatus($"Generated {_frameBuffer.Count} animated frames");
            }
            else
            {
                UpdateStatus("No frames were generated");
            }
            
            _isGenerating = false;
            SetButtonsEnabled(true);
            if (progressSlider != null) progressSlider.value = 1f;
        }
        
        IEnumerator GenerateTalkingHead()
        {
            if (!ValidateInputsForAnimatedOnly()) yield break;
            
            UpdateStatus("Generating talking head...");
            SetButtonsEnabled(false);
            
            // Clear previous results and reset state
            ClearFrameBuffer();
            _currentPreviewFrame = 0;
            _isGenerating = true;
            _lastFrameTime = Time.time;
            _isPlayingAudio = true;
            
            // Start frame generation in background
            _frameGenerationCoroutine = StartCoroutine(GenerateTalkingHeadFramesAsync());
            
            // Start preview animation immediately (will show frames as they become available)
            StartStreamingPreviewAnimation();
            
            // Wait for generation to complete
            while (_isGenerating)
            {
                yield return null;
            }
            
            // Final status update
            if (_frameBuffer.Count > 0)
            {
                UpdateStatus($"Generated {_frameBuffer.Count} animated frames");
            }
            else
            {
                UpdateStatus("No frames were generated");
            }
            
            _isGenerating = false;
            SetButtonsEnabled(true);
            if (progressSlider != null) progressSlider.value = 1f;
        }
        
        IEnumerator GenerateFramesAsyncFromVideo()
        {
            int frameCount = 0;
            
            // Clear video frame buffer if saving video
            if (saveVideoOutput)
            {
                UpdateStatus("Video saving enabled - preparing output...");
                
                // Setup output directory upfront
                try
                {
                    var path = videoOutputPath;
                    if (videoPlayer != null && videoPlayer.clip != null)
                    {
                        path = videoPlayer.clip.name;
                    }
                    _outputDirectory = System.IO.Path.Combine(Application.streamingAssetsPath, path);
                    Debug.Log($"[LiveTalkDemo] Output directory: {_outputDirectory}");
                    _savedFrameCount = 0;
                    
                    // Create output directory if it doesn't exist
                    if (!System.IO.Directory.Exists(_outputDirectory))
                    {
                        System.IO.Directory.CreateDirectory(_outputDirectory);
                    }
                    
                    UpdateStatus("Output directory ready - starting generation...");
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"[LiveTalkDemo] Error setting up video output directory: {e.Message}");
                    UpdateStatus("Error setting up video output - continuing without saving");
                    saveVideoOutput = false; // Disable saving for this session
                }
            }
            
            // Call the API to get the frame enumerator
            var stream = _api.GenerateAnimatedTexturesAsync(sourceImage, videoPlayer);
            _totalExpectedFrames = stream.TotalExpectedFrames;

            // Process frames as they become available
            while (stream.HasMoreFrames)
            {
                var awaiter = stream.WaitForNext();
                yield return awaiter;                // blocks until *one* texture is ready

                if (awaiter.Texture != null)
                {
                    // Add frame to buffer on main thread
                    AddFrameToBuffer(awaiter.Texture);
                    
                    frameCount++;
                    
                    // Update progress
                    if (progressSlider != null && _totalExpectedFrames > 0)
                    {
                        progressSlider.value = (float)frameCount / _totalExpectedFrames;
                    }
                    
                    string statusMessage = $"🎨 Generated frame {frameCount}/{_totalExpectedFrames}...";
                    if (saveVideoOutput)
                    {
                        statusMessage += $" (saved {_savedFrameCount} PNG)";
                    }
                    UpdateStatus(statusMessage);
                }
                
                yield return null; // Yield control to allow other coroutines and UI to update
            }
            
            if (frameCount > 0)
            {
                string completionMessage = $"✓ Frame generation completed: {frameCount} frames";
                if (saveVideoOutput)
                {
                    completionMessage += $" ({_savedFrameCount} PNG frames saved to {System.IO.Path.GetFileName(_outputDirectory)})";
                    Debug.Log($"[LiveTalkDemo] You can use external tools like FFmpeg to combine frames into video:");
                    Debug.Log($"ffmpeg -framerate 25 -i \"{System.IO.Path.Combine(_outputDirectory, "frame_%06d.png")}\" -c:v libx264 -pix_fmt yuv420p \"{System.IO.Path.Combine(_outputDirectory, "video.mp4")}\"");
                }
                UpdateStatus(completionMessage);
            }
            else
            {
                UpdateStatus("✗ No frames were generated");
            }
            
            _isGenerating = false;
        }

        /// <summary>
        /// Generate frames asynchronously and add them to buffer as they become available
        /// </summary>
        IEnumerator GenerateFramesAsync()
        {
            int frameCount = 0;
            
            // Call the API to get the frame enumerator
            var stream = _api.GenerateAnimatedTexturesAsync(sourceImage, drivingFramesFolderPath, 50);
            _totalExpectedFrames = stream.TotalExpectedFrames;

            // Process frames as they become available
            while (stream.HasMoreFrames)
            {
                var awaiter = stream.WaitForNext();
                yield return awaiter;                // blocks until *one* texture is ready

                if (awaiter.Texture != null)
                {
                    // Add frame to buffer on main thread
                    AddFrameToBuffer(awaiter.Texture);
                    frameCount++;
                    
                    // Update progress
                    if (progressSlider != null && _totalExpectedFrames > 0)
                    {
                        progressSlider.value = (float)frameCount / _totalExpectedFrames;
                    }
                    
                    UpdateStatus($"Generated frame {frameCount}/{_totalExpectedFrames}...");
                }
                
                yield return null; // Yield control to allow other coroutines and UI to update
            }
            
            if (frameCount > 0)
            {
                UpdateStatus($"Frame generation completed: {frameCount} frames");
            }
            else
            {
                UpdateStatus("No frames were generated");
            }
            
            _isGenerating = false;
        }
        
        IEnumerator GenerateTalkingHeadFramesAsync()
        {
            int frameCount = 0;
            
            // Call the API to get the frame enumerator
            var stream = _api.GenerateTalkingHeadAsync(sourceImage, talkingHeadFolderPath, audioClip);
            _totalExpectedFrames = stream.TotalExpectedFrames;

            // Process frames as they become available
            while (stream.HasMoreFrames || frameCount < _totalExpectedFrames)
            {
                var awaiter = stream.WaitForNext();
                yield return awaiter;                // blocks until *one* texture is ready
                _totalExpectedFrames = stream.TotalExpectedFrames;

                if (awaiter.Texture != null)
                {
                    // Add frame to buffer on main thread
                    AddFrameToBuffer(awaiter.Texture);
                    frameCount++;
                    
                    // Update progress
                    if (progressSlider != null && _totalExpectedFrames > 0)
                    {
                        progressSlider.value = (float)frameCount / _totalExpectedFrames;
                    }
                    
                    UpdateStatus($"Generated frame {frameCount}/{_totalExpectedFrames}...");
                }
                
                yield return null; // Yield control to allow other coroutines and UI to update
            }
            
            if (frameCount > 0)
            {
                UpdateStatus($"Frame generation completed: {frameCount} frames");
            }
            else
            {
                UpdateStatus("No frames were generated");
            }
            
            _isGenerating = false;
        }
        
        /// <summary>
        /// Create a character using the Character Creator API
        /// </summary>
        IEnumerator CreateCharacter()
        {
            if (!ValidateInputsForCharacterCreation()) yield break;
            
            UpdateStatus("Creating character...");
            SetButtonsEnabled(false);
            
            UpdateStatus($"Creating character '{characterName}' with {characterGender} voice...");
            
            // Track completion state
            bool completed = false;
            Character createdCharacter = null;
            System.Exception error = null;
            
            // Create character asynchronously with callbacks
            yield return _api.CreateCharacterAsync(
                characterName,
                characterGender,
                sourceImage,
                characterPitch,
                characterSpeed,
                characterIntro,
                null,
                onComplete: (character) => {
                    createdCharacter = character;
                    completed = true;
                },
                onError: (ex) => {
                    error = ex;
                },
                creationMode: CreationMode.SingleExpression
            );
            
            // Handle the result
            if (error != null && !completed)
            {
                UpdateStatus($"Character creation failed: {error.Message}");
                Debug.LogError($"[LiveTalkDemo] Character creation error: {error}");
            }
            else if (createdCharacter != null)
            {
                UpdateStatus($"Character '{createdCharacter.Name}' created successfully!");
                Debug.Log($"[LiveTalkDemo] Character saved to: {LiveTalkAPI.CharacterSaveLocation}");
                Debug.Log($"[LiveTalkDemo] Character details: {createdCharacter.Name}, {createdCharacter.Gender}, Pitch: {createdCharacter.Pitch}, Speed: {createdCharacter.Speed}");
                
                // Character created - you can now use its ID to load it
            }
            else
            {
                UpdateStatus("Character creation completed but no character was returned");
            }
            
            SetButtonsEnabled(true);
        }
        
        /// <summary>
        /// Load a character using the character ID from inspector
        /// </summary>
        IEnumerator LoadCharacter()
        {
            if (string.IsNullOrEmpty(characterIdToLoad))
            {
                UpdateStatus("Please set a character ID to load in the inspector");
                yield break;
            }
            
            UpdateStatus($"Loading character: {characterIdToLoad}...");
            SetButtonsEnabled(false);
            
            // Track completion state
            bool completed = false;
            Character loadedCharacter = null;
            System.Exception error = null;
            
            // Load character asynchronously
            yield return _api.LoadCharacterAsyncFromId(
                characterIdToLoad,
                onComplete: (character) => {
                    loadedCharacter = character;
                    completed = true;
                },
                onError: (ex) => {
                    error = ex;
                }
            );
            
            // Handle the result
            if (error != null && !completed)
            {
                UpdateStatus($"Failed to load character: {error.Message}");
                Debug.LogError($"[LiveTalkDemo] Character loading error: {error}");
            }
            else if (loadedCharacter != null)
            {
                _loadedCharacter = loadedCharacter;
                UpdateStatus($"Character '{loadedCharacter.Name}' loaded successfully! Ready to speak.");
                Debug.Log($"[LiveTalkDemo] Character loaded: {loadedCharacter.Name}, Data loaded: {loadedCharacter.IsDataLoaded}");
            }
            else
            {
                UpdateStatus("Character loading completed but no character was returned");
            }
            
            SetButtonsEnabled(true);
        }
        
        /// <summary>
        /// Make the loaded character speak
        /// </summary>
        IEnumerator SpeakWithLoadedCharacter()
        {
            _overrideAudio = true;
            if (_loadedCharacter == null)
            {
                UpdateStatus("No character loaded. Please load a character first.");
                yield break;
            }
            
            if (!_loadedCharacter.IsDataLoaded)
            {
                UpdateStatus("Character data not loaded. Please reload the character.");
                yield break;
            }
            
            if (string.IsNullOrEmpty(textToSpeak))
            {
                UpdateStatus("Please set text to speak in the inspector");
                yield break;
            }
            
            UpdateStatus($"Making {_loadedCharacter.Name} speak: \"{textToSpeak}\"...");
            SetButtonsEnabled(false);
            
            // Clear previous results and reset state
            ClearFrameBuffer();
            _currentPreviewFrame = 0;
            _isGenerating = true;
            _lastFrameTime = Time.time;
            _isPlayingAudio = true;
            
            // Generate speech with the loaded character
            FrameStream speechStream = null;
            System.Exception speechError = null;
            yield return _loadedCharacter.SpeakAsync(
                textToSpeak,
                expressionIndex: generateVoiceOnly ? -1 : 0, // Use talk-neutral expression
                onComplete: (stream, audioClip) => {
                    speechStream = stream;
                    audioSource.clip = audioClip;
                },
                onError: (ex) => {
                    speechError = ex;
                }
            );
            
            if (speechError != null)
            {
                UpdateStatus($"Speech generation failed: {speechError.Message}");
                Debug.LogError($"[LiveTalkDemo] Speech error: {speechError}");
                _isGenerating = false;
                SetButtonsEnabled(true);
                yield break;
            }
            
            if (speechStream == null)
            {
                UpdateStatus("Speech generation completed but no stream was returned");
                _isGenerating = false;
                SetButtonsEnabled(true);
                yield break;
            }

            if (generateVoiceOnly)
            {
                _isGenerating = false;
                SetButtonsEnabled(true);
                PlayAudio();
                yield break;
            }
            
            // Start processing the speech stream
            _totalExpectedFrames = speechStream.TotalExpectedFrames;
            
            // Start preview animation
            StartStreamingPreviewAnimation();
            
            // Process speech frames as they become available
            int frameCount = 0;
            while (speechStream.HasMoreFrames)
            {
                var awaiter = speechStream.WaitForNext();
                yield return awaiter;
                
                if (awaiter.Texture != null)
                {
                    AddFrameToBuffer(awaiter.Texture);
                    frameCount++;
                    
                    // Update progress
                    if (progressSlider != null && _totalExpectedFrames > 0)
                    {
                        progressSlider.value = (float)frameCount / _totalExpectedFrames;
                    }
                    
                    UpdateStatus($"{_loadedCharacter.Name} speaking... frame {frameCount}/{_totalExpectedFrames}");
                }
                
                yield return null;
            }
            
            UpdateStatus($"{_loadedCharacter.Name} finished speaking! Generated {frameCount} frames.");
            _isGenerating = false;
            SetButtonsEnabled(true);
            if (progressSlider != null) progressSlider.value = 1f;
        }
        

        
        /// <summary>
        /// Add a new frame to the buffer (thread-safe) and save as PNG if video output is enabled
        /// </summary>
        void AddFrameToBuffer(Texture2D frame)
        {
            if (frame != null)
            {
                _frameBuffer.Add(frame);
                
                // Save frame as PNG immediately if video output is enabled
                if (saveVideoOutput && !string.IsNullOrEmpty(_outputDirectory))
                {
                    try
                    {
                        byte[] pngData = frame.EncodeToPNG();
                        string framePath = System.IO.Path.Combine(_outputDirectory, $"frame_{_savedFrameCount:D6}.png");
                        System.IO.File.WriteAllBytes(framePath, pngData);
                        _savedFrameCount++;
                    }
                    catch (System.Exception e)
                    {
                        Debug.LogError($"[LiveTalkDemo] Error saving frame {_savedFrameCount} as PNG: {e.Message}");
                    }
                }
            }
        }
        

        
        /// <summary>
        /// Start streaming preview animation that plays at 25FPS
        /// </summary>
        void StartStreamingPreviewAnimation()
        {
            if (_previewCoroutine != null)
                StopCoroutine(_previewCoroutine);
                
            if (previewImage != null)
            {
                _previewCoroutine = StartCoroutine(StreamingPreviewAnimation());
            }
        }
        
        /// <summary>
        /// Preview animation that plays at 25FPS, only looping once all expected frames are generated
        /// </summary>
        IEnumerator StreamingPreviewAnimation()
        {
            _currentPreviewFrame = 0;
            _lastFrameTime = Time.time;
            
            // FPS tracking variables
            float fpsUpdateInterval = 0.5f; // Update FPS display every 0.5 seconds
            float lastFpsUpdateTime = Time.time;
            int framesSinceLastFpsUpdate = 0;
            float currentFps = 0f;

            
            while (true)
            {
                // Check if we have frames to display
                if (_frameBuffer.Count > 0)
                {
                    // Determine if we should loop or pause
                    bool allFramesGenerated = !_isGenerating || (_totalExpectedFrames > 0 && _frameBuffer.Count >= _totalExpectedFrames);
                    
                    // Ensure we don't go beyond available frames
                    if (_currentPreviewFrame >= _frameBuffer.Count)
                    {
                        if (allFramesGenerated)
                        {
                            // All frames generated - loop back to start
                            _currentPreviewFrame = 0;
                            PlayAudio();
                        }
                        else
                        {
                            // Still generating - pause at last frame
                            _currentPreviewFrame = _frameBuffer.Count - 1;
                        }
                    }
                    
                    // Display current frame
                    if (_currentPreviewFrame < _frameBuffer.Count && previewImage != null)
                    {
                        previewImage.texture = _frameBuffer[_currentPreviewFrame];
                    }
                    
                    // Check if it's time for the next frame (25 FPS)
                    float currentTime = Time.time;
                    if (currentTime - _lastFrameTime >= FRAME_INTERVAL)
                    {
                        bool frameAdvanced = false;
                        
                        if (allFramesGenerated)
                        {
                            // All frames ready - advance and loop
                            _currentPreviewFrame++;
                            _lastFrameTime = currentTime;
                            frameAdvanced = true;
                            
                            // Loop back to start if at end
                            if (_currentPreviewFrame >= _frameBuffer.Count)
                            {
                                _currentPreviewFrame = 0;
                                PlayAudio();
                            }
                        }
                        else
                        {
                            // Still generating - only advance if next frame is available
                            if (_currentPreviewFrame + 1 < _frameBuffer.Count)
                            {
                                _currentPreviewFrame++;
                                _lastFrameTime = currentTime;
                                frameAdvanced = true;
                            }
                            // Else: pause at current frame until next frame is available
                        }
                        
                        // Update FPS tracking when frame actually advances
                        if (frameAdvanced)
                        {
                            framesSinceLastFpsUpdate++;
                            
                            // Update FPS display periodically
                            if (currentTime - lastFpsUpdateTime >= fpsUpdateInterval)
                            {
                                currentFps = framesSinceLastFpsUpdate / (currentTime - lastFpsUpdateTime);
                                if (fpsText != null)
                                {
                                    fpsText.text = $"FPS: {currentFps:F1}";
                                }
                                
                                // Reset FPS tracking
                                framesSinceLastFpsUpdate = 0;
                                lastFpsUpdateTime = currentTime;
                            }
                        }
                    }
                }
                else if (!_isGenerating)
                {
                    // No frames and not generating - break out of loop
                    break;
                }
                
                yield return null; // Wait for next frame
            }
        }

        void PlayAudio()
        {
            if (audioSource != null && _overrideAudio && _isPlayingAudio)
            {
                audioSource.Play();
            }
            else if (audioSource != null && audioClip != null && _isPlayingAudio)
            {
                if (audioSource.clip != audioClip)
                {
                    audioSource.clip = audioClip;
                }
                audioSource.Play();
            }
        }
        
        /// <summary>
        /// Clear the frame buffer and clean up textures
        /// </summary>
        void ClearFrameBuffer()
        {
            // Clean up existing frames
            foreach (var frame in _frameBuffer)
            {
                if (frame != null)
                {
                    DestroyImmediate(frame);
                }
            }
            _frameBuffer.Clear();
        }
        
        /// <summary>
        /// Validate inputs for animated textures only
        /// </summary>
        bool ValidateInputsForAnimatedOnly()
        {
            if (sourceImage == null)
            {
                UpdateStatus("Please assign a source image");
                return false;
            }
            
            return true;
        }
        
        /// <summary>
        /// Validate inputs for character creation
        /// </summary>
        bool ValidateInputsForCharacterCreation()
        {
            if (sourceImage == null)
            {
                UpdateStatus("Please assign a source image for character creation");
                return false;
            }
            
            if (string.IsNullOrEmpty(characterName))
            {
                UpdateStatus("Please provide a character name");
                return false;
            }
            
            if (string.IsNullOrEmpty(LiveTalkAPI.CharacterSaveLocation))
            {
                UpdateStatus("Please provide a save location for characters");
                return false;
            }
            
            return true;
        }
        
        /// <summary>
        /// Update status text
        /// </summary>
        void UpdateStatus(string message)
        {
            if (statusText != null)
                statusText.text = message;
                
            Debug.Log($"[LiveTalkDemo] {message}");
        }
        
        /// <summary>
        /// Enable/disable UI buttons
        /// </summary>
        void SetButtonsEnabled(bool enabled)
        {
            if (generateButton != null)
                generateButton.interactable = enabled;
                
            if (generateTalkingHeadButton != null)
                generateTalkingHeadButton.interactable = enabled;
                
            if (createCharacterButton != null)
                createCharacterButton.interactable = enabled;
                
            if (loadCharacterButton != null)
                loadCharacterButton.interactable = enabled;
                
            if (speakButton != null)
                speakButton.interactable = enabled && _loadedCharacter != null && _loadedCharacter.IsDataLoaded;
        }
        
        void OnDestroy()
        {
            if (_previewCoroutine != null)
                StopCoroutine(_previewCoroutine);
                
            if (_frameGenerationCoroutine != null)
                StopCoroutine(_frameGenerationCoroutine);
                
            // Clean up frame buffer
            ClearFrameBuffer();
            _api?.Dispose();
        }
    }
}
