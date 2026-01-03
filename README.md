# LiveTalk-Unity

Unity package for using LiveTalk on-device models for real-time talking head generation and character animation.

## What is LiveTalk?

LiveTalk is a unified, high-performance talking head generation system that combines the power of [LivePortrait](https://github.com/KwaiVGI/LivePortrait) and [MuseTalk](https://github.com/TMElyralab/MuseTalk) open-source repositories. The PyTorch models from these projects have been ported to ONNX format and optimized for CoreML to enable efficient on-device inference in Unity.

**LivePortrait** provides facial animation and expression transfer capabilities, while **MuseTalk** handles real-time lip synchronization with audio. Together, they create a complete pipeline for generating natural-looking talking head videos from avatar images and audio input. 
[Spark-TTS-Unity](https://github.com/arghyasur1991/Spark-TTS-Unity) is the dependency package for TTS generation

## Key Features

* 🎮 **Unity-Native Integration**: Complete API designed specifically for Unity with singleton pattern
* 🎭 **Dual-Pipeline Processing**: LivePortrait for facial animation + MuseTalk for lip sync
* 👤 **Advanced Character System**: Create, save, and load characters with multiple expressions and voices
* 💻 **Runs Offline**: All processing happens on-device with ONNX Runtime
* ⚡ **Real-time Performance**: Optimized for real-time inference with frame streaming
* 🎨 **Multiple Expression Support**: 7 built-in expressions (talk-neutral, approve, disapprove, smile, sad, surprised, confused)
* 🔊 **Integrated TTS**: Built-in SparkTTS integration for voice generation
* 📦 **Cross-Platform Character Format**: Supports both folder and macOS bundle formats
* 🎥 **Flexible Input**: Supports images, videos, and directory-based driving frames
* 🧠 **Memory Management**: Configurable memory usage modes for desktop and mobile devices
* 🎭 **Flexible Creation Modes**: Voice-only, single expression, or full character creation

## Perfect For

* AI-driven NPCs in games
* Virtual assistants and chatbots
* Real-time character animation
* Interactive storytelling applications
* Video content generation
* Accessibility features
* Virtual avatars and digital humans

## Installation

### Using Unity Package Manager (Recommended)

1. Open your Unity project
2. Open the Package Manager (Window > Package Manager)
3. Click the "+" button in the top-left corner
4. Select "Add package from git URL..."
5. Enter the repository URL: `https://github.com/arghyasur1991/LiveTalk-Unity.git`
6. Click "Add"

### Manual Installation

1. Clone this repository
2. Copy the contents into your Unity project's Packages folder

## Dependencies

This package requires the following Unity packages:
- com.genesis.sparktts.unity

### Setting up Package Dependencies

Some dependencies require additional scoped registry configuration. Add the following to your project's `Packages/manifest.json` file:

```json
{
  "scopedRegistries": [
    {
      "name": "NPM",
      "url": "https://registry.npmjs.com",
      "scopes": [
        "com.github.asus4"
      ]
    }
  ],
  "dependencies": {
    "com.genesis.LiveTalk.unity": "https://github.com/arghyasur1991/LiveTalk-Unity.git",
    // ... other dependencies
  }
}
```

**Note**: The git URL `https://github.com/arghyasur1991/LiveTalk-Unity.git` will automatically fetch the latest version of the package.

## Model Setup

### LiveTalk Models

LiveTalk requires ONNX models from both LivePortrait and MuseTalk in the following location:

```
Assets/StreamingAssets/LiveTalk/
  └── models/
      ├── LivePortrait/
      │   ├── *.onnx
      └── MuseTalk/
          ├── *.onnx
```

### SparkTTS Models

SparkTTS models are required for voice generation and should be placed in:

```
Assets/StreamingAssets/SparkTTS/
  ├── *.onnx
  └── LLM/
      ├── model.onnx
      ├── model.onnx_data
      ├── ...
```

### Model Deployment Tool

LiveTalk includes a built-in Editor tool that automatically analyzes your codebase and copies only the required models from `Assets/Models` to `StreamingAssets` with the correct precision settings (FP16, FP32, etc.).

**Access the tool**: `Window > LiveTalk > Model Deployment Tool`

#### Key Features

* **Precision-Aware**: Copies only the required precision variants (FP16/FP32) based on code analysis
* **Size Optimization**: Reduces build size by excluding unused models
* **Folder Structure Preservation**: Maintains the correct directory structure in StreamingAssets
* **Backup Support**: Creates backups of existing models before overwriting
* **Dry Run Mode**: Preview changes without actually copying files

#### How to Use

1. **Open the tool**: Go to `Window > LiveTalk > Model Deployment Tool`
2. **Configure paths**: 
   - Source: `Assets/Models` (automatically detected)
   - Destination: `Assets/StreamingAssets/LiveTalk` (automatically configured)
3. **Select components**: Choose which model categories to deploy:
   - ✅ **SparkTTS Models** (deployed via SparkTTS-Unity package)
   - ✅ **LivePortrait Models** (deployed directly)
   - ✅ **MuseTalk Models** (deployed directly)
4. **Review selection**: The tool shows you exactly which LiveTalk models will be copied and their file sizes
5. **Deploy**: Click "Deploy All Models" to copy both LiveTalk and SparkTTS models using their respective deployment systems

#### Model Precision Settings

The tool selects the used precision for each model based on the LiveTalk codebase:

| Model Category | Precision | Execution Provider | Notes |
|---|---|---|---|
| **LivePortrait** | | | |
| warping_spade | FP16 | CoreML | GPU-accelerated rendering |
| Other LivePortrait | FP32 | CoreML | Full precision for facial features |
| **MuseTalk** | | | |
| unet, vae_encoder, vae_decoder | FP16 | CoreML | GPU-accelerated inference |
| whisper_encoder, positional_encoding | FP32 | CPU | Audio processing precision |
| **SparkTTS** | | | |
| *Models deployed via SparkTTS-Unity package* | *See SparkTTS documentation* | *Various* | *Handled by SparkTTS deployment tool* |

#### Advanced Options

* **Overwrite Existing**: Replace existing models in StreamingAssets
* **Create Backup**: Keep .backup copies of replaced files (includes .onnx.data files)
* **Dry Run**: Preview operations without copying files

#### Large Model Handling

The tool automatically handles large models that use separate data files:
- **MuseTalk UNet**: `unet.onnx` (710KB) + `unet.onnx.data` (3.2GB) - uses dot notation
- **SparkTTS LLM**: Handled by SparkTTS-Unity deployment tool with `model.onnx_data` files

LiveTalk model and data files are copied together and included in size calculations and backup operations. SparkTTS models are handled by the SparkTTS-Unity package's own deployment system.

This tool ensures your Unity project includes only the models you actually need, significantly reducing build size while maintaining optimal performance.

#### Standalone SparkTTS Deployment

SparkTTS models can also be deployed independently using the SparkTTS-Unity package's standalone tool:

**Access**: `Window > SparkTTS > Model Deployment Tool`

This allows you to:
- Deploy only SparkTTS models without LiveTalk models
- Use SparkTTS in projects that don't include LiveTalk
- Have fine-grained control over SparkTTS model deployment

### Downloading Pre-Exported Models

#### LiveTalk Models
Download the pre-exported ONNX models from [Google Drive](https://drive.google.com/file/d/1UvssShqniAj_p-yw0dLDTWQEqe-O_n6K/view?usp=sharing).

1. Download the ZIP file from the link
2. Extract the contents
3. Copy the extracted `LiveTalk` folder with models to your Unity project's `Assets/Models/` directory
4. **Use the Model Deployment Tool** (recommended): Go to `Window > LiveTalk > Model Deployment Tool` to automatically copy only the required models with optimal precision settings

#### SparkTTS Models
Check the [Model Setup](https://github.com/arghyasur1991/Spark-TTS-Unity#model-setup) section of Spark-TTS-Unity

### Exporting Models (Coming Soon)

Coming Soon - conversion scripts to export models from the original Python repositories:

- **LivePortrait**: [https://github.com/KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- **MuseTalk**: [https://github.com/TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)

The export scripts will convert PyTorch models to ONNX format and apply CoreML optimizations for Unity integration.

## Usage

### Initialization and Model Loading

```csharp
using UnityEngine;
using LiveTalk.API;
using System.Threading.Tasks;

public class LiveTalkExample : MonoBehaviour
{
    async void Start()
    {
        // Initialize the LiveTalk system
        LiveTalkAPI.Instance.Initialize(
            logLevel: LogLevel.INFO,
            characterSaveLocation: "", // Uses default location (persistentDataPath/Characters)
            parentModelPath: "",       // Uses StreamingAssets
            memoryUsage: MemoryUsage.Performance // Load all models at startup
        );
        
        // Wait for all models to be loaded (Performance mode)
        // This includes LivePortrait, MuseTalk, and SparkTTS models
        await LiveTalkAPI.Instance.WaitForAllModelsAsync(
            onProgress: (modelName, progress) => {
                Debug.Log($"Loading {modelName}: {progress * 100:F0}%");
            }
        );
        
        Debug.Log("All models loaded and ready!");
        
        // Now you can create characters and generate speech...
    }
}
```

#### Model Loading Progress (Performance Mode)

When using `MemoryUsage.Performance` mode, you can track model loading progress:

```csharp
// Simple usage - just wait for completion
await LiveTalkAPI.Instance.WaitForAllModelsAsync();

// With progress tracking
await LiveTalkAPI.Instance.WaitForAllModelsAsync(
    onProgress: (modelName, progress) => {
        // progress is 0.0 to 1.0 for each model group
        // modelName will be: "LivePortrait Animation", "MuseTalk Animation", or "Voice Synthesis"
        UpdateLoadingUI($"Loading {modelName}", progress);
    }
);
```

**Note**: Model loading is only necessary in `MemoryUsage.Performance` mode. In `Balanced` and `Optimal` modes, models load on-demand during first use.

### Basic Setup

```csharp
using UnityEngine;
using LiveTalk.API;

public class LiveTalkExample : MonoBehaviour
{
    void Start()
    {
        // Initialize the LiveTalk system
        LiveTalkAPI.Instance.Initialize(
            logLevel: LogLevel.INFO,
            characterSaveLocation: "", // Uses default location (persistentDataPath/Characters)
            parentModelPath: "",       // Uses StreamingAssets
            memoryUsage: MemoryUsage.Balanced // Recommended for desktop
        );
    }
}
```

### Memory Usage Modes

LiveTalk supports different memory usage configurations optimized for various device types:

```csharp
// For desktop devices (default) - balanced memory and performance
LiveTalkAPI.Instance.Initialize(memoryUsage: MemoryUsage.Balanced);

// For performance-critical applications - loads all models upfront
LiveTalkAPI.Instance.Initialize(memoryUsage: MemoryUsage.Performance);

// For mobile devices - minimal memory footprint
LiveTalkAPI.Instance.Initialize(memoryUsage: MemoryUsage.Optimal);

// Automatic selection based on platform
MemoryUsage memoryUsage = Application.isMobilePlatform 
    ? MemoryUsage.Optimal 
    : MemoryUsage.Balanced;
LiveTalkAPI.Instance.Initialize(memoryUsage: memoryUsage);
```

### Character Creation

```csharp
using UnityEngine;
using LiveTalk.API;
using System.Collections;

public class CharacterCreation : MonoBehaviour
{
    [SerializeField] private Texture2D characterImage;
    
    IEnumerator Start()
    {
        // Initialize API
        LiveTalkAPI.Instance.Initialize();
        
        // Create a new character with all expressions
        yield return LiveTalkAPI.Instance.CreateCharacterAsync(
            name: "MyCharacter",
            gender: Gender.Female,
            image: characterImage,
            pitch: Pitch.Moderate,
            speed: Speed.Moderate,
            intro: "Hello, I am your virtual assistant!",
            voicePromptPath: null, // Generate voice from style parameters
            onComplete: (character) => {
                Debug.Log($"Character created: {character.Name}");
            },
            onError: (error) => {
                Debug.LogError($"Character creation failed: {error.Message}");
            }
        );
    }
}
```

### Character Creation Modes

LiveTalk supports different creation modes for flexibility:

```csharp
// Voice Only - fastest, no visual expressions generated
yield return LiveTalkAPI.Instance.CreateCharacterAsync(
    name: "VoiceOnlyCharacter",
    gender: Gender.Male,
    image: null, // No image needed for voice-only
    pitch: Pitch.Low,
    speed: Speed.Moderate,
    intro: "Hello!",
    voicePromptPath: null,
    onComplete: OnCharacterCreated,
    onError: OnError,
    creationMode: CreationMode.VoiceOnly
);

// Single Expression - voice + talk-neutral expression only
yield return LiveTalkAPI.Instance.CreateCharacterAsync(
    name: "QuickCharacter",
    gender: Gender.Female,
    image: characterImage,
    pitch: Pitch.Moderate,
    speed: Speed.Moderate,
    intro: "Hello!",
    voicePromptPath: null,
    onComplete: OnCharacterCreated,
    onError: OnError,
    creationMode: CreationMode.SingleExpression
);

// All Expressions - full character with all 7 expressions (default)
yield return LiveTalkAPI.Instance.CreateCharacterAsync(
    name: "FullCharacter",
    gender: Gender.Female,
    image: characterImage,
    pitch: Pitch.High,
    speed: Speed.High,
    intro: "Hello!",
    voicePromptPath: null,
    onComplete: OnCharacterCreated,
    onError: OnError,
    creationMode: CreationMode.AllExpressions
);
```

### Using a Voice Reference

You can create a character voice from an existing audio reference instead of generating from style parameters:

```csharp
// Create character with voice cloned from reference audio
yield return LiveTalkAPI.Instance.CreateCharacterAsync(
    name: "ClonedVoiceCharacter",
    gender: Gender.Female,
    image: characterImage,
    pitch: Pitch.Moderate,  // These are ignored when voicePromptPath is provided
    speed: Speed.Moderate,
    intro: "Hello!",
    voicePromptPath: "/path/to/reference_voice.wav", // Clone voice from this audio
    onComplete: OnCharacterCreated,
    onError: OnError
);
```

### Character Loading and Speech

```csharp
using UnityEngine;
using LiveTalk.API;
using System.Collections;

public class CharacterSpeech : MonoBehaviour
{
    private Character loadedCharacter;
    
    IEnumerator Start()
    {
        // Initialize API
        LiveTalkAPI.Instance.Initialize();
        
        // Load an existing character by ID
        string characterId = "your-character-id";
        yield return LiveTalkAPI.Instance.LoadCharacterAsyncFromId(
            characterId,
            onComplete: (character) => {
                loadedCharacter = character;
                Debug.Log($"Character loaded: {character.Name}");
                
                // Make the character speak
                StartCoroutine(MakeCharacterSpeak());
            },
            onError: (error) => {
                Debug.LogError($"Character loading failed: {error.Message}");
            }
        );
    }
    
    IEnumerator MakeCharacterSpeak()
    {
        if (loadedCharacter == null) yield break;
        
        // Speak with lip sync animation
        // Two callbacks: onAudioReady (when audio is ready), onAnimationComplete (when all frames are done)
        // Audio and animation generation are pipelined - audio for next segment can generate while current animates
        yield return loadedCharacter.StartSpeakWithCallbacks(
            text: "Hello! I can speak with realistic lip sync!",
            expressionIndex: 0, // Use talk-neutral expression
            onAudioReady: (frameStream, audioClip) => {
                // Called as soon as audio is ready - you can schedule next speech here
                // frameStream will receive animation frames as they're generated
                Debug.Log($"Audio ready, expecting {frameStream.TotalExpectedFrames} frames");
                StartCoroutine(PlayGeneratedVideo(frameStream, audioClip));
            },
            onAnimationComplete: () => {
                // Called when all animation frames have been generated
                Debug.Log("Animation generation complete");
            },
            onError: (error) => {
                Debug.LogError($"Speech generation failed: {error.Message}");
            }
        );
    }
    
    IEnumerator PlayGeneratedVideo(FrameStream frameStream, AudioClip audioClip)
    {
        // Play the audio
        GetComponent<AudioSource>().clip = audioClip;
        GetComponent<AudioSource>().Play();
        
        // Process video frames as they arrive
        while (frameStream.HasMoreFrames)
        {
            var frameAwaiter = frameStream.WaitForNext();
            yield return frameAwaiter;
            
            if (frameAwaiter.Texture != null)
            {
                // Display the frame (e.g., on a RawImage component)
                GetComponent<UnityEngine.UI.RawImage>().texture = frameAwaiter.Texture;
            }
        }
    }
}
```

### Voice-Only Speech (No Animation)

For scenarios where you only need audio without video frames:

```csharp
// Generate voice only (no lip sync animation)
yield return loadedCharacter.StartSpeakWithCallbacks(
    text: "This is voice-only output!",
    expressionIndex: -1, // -1 means voice only, no video frames
    onAudioReady: (frameStream, audioClip) => {
        // frameStream will be empty, only audioClip is populated
        GetComponent<AudioSource>().clip = audioClip;
        GetComponent<AudioSource>().Play();
    },
    onAnimationComplete: null, // Optional - not needed for voice-only
    onError: (error) => {
        Debug.LogError($"Speech generation failed: {error.Message}");
    }
);
```

### Pipelined Speech Generation

The `StartSpeakWithCallbacks` method enables pipelined audio and animation processing:

```csharp
// Start first speech
yield return character.StartSpeakWithCallbacks(
    text: "First sentence.",
    expressionIndex: 0,
    onAudioReady: (stream1, audio1) => {
        PlayAudioAndAnimation(stream1, audio1);
        
        // As soon as audio is ready, schedule next speech
        // Audio for segment 2 generates while segment 1 animates
        StartCoroutine(character.StartSpeakWithCallbacks(
            text: "Second sentence immediately after.",
            expressionIndex: 0,
            onAudioReady: (stream2, audio2) => {
                EnqueueAudioAndAnimation(stream2, audio2);
            },
            onAnimationComplete: null,
            onError: OnError
        ));
    },
    onAnimationComplete: null,
    onError: OnError
);
```

This pipelining significantly improves responsiveness by overlapping audio generation for the next segment with animation generation for the current segment.

### Loading Character from Path

You can also load a character directly from a file path:

```csharp
// Load character from a specific path
yield return LiveTalkAPI.Instance.LoadCharacterAsyncFromPath(
    "/path/to/character/folder",
    onComplete: (character) => {
        Debug.Log($"Character loaded: {character.Name}");
    },
    onError: (error) => {
        Debug.LogError($"Loading failed: {error.Message}");
    }
);
```

### Facial Animation (LivePortrait Only)

```csharp
using UnityEngine;
using LiveTalk.API;
using System.Collections;
using UnityEngine.Video;

public class FacialAnimation : MonoBehaviour
{
    [SerializeField] private Texture2D sourceImage;
    [SerializeField] private VideoPlayer drivingVideo;
    
    IEnumerator Start()
    {
        // Initialize API
        LiveTalkAPI.Instance.Initialize();
        
        // Generate animated textures using LivePortrait
        var animationStream = LiveTalkAPI.Instance.GenerateAnimatedTexturesAsync(
            sourceImage, 
            drivingVideo, 
            maxFrames: -1 // Process all frames
        );
        
        // Process the animated frames
        while (animationStream.HasMoreFrames)
        {
            var frameAwaiter = animationStream.WaitForNext();
            yield return frameAwaiter;
            
            if (frameAwaiter.Texture != null)
            {
                // Display the animated frame
                GetComponent<UnityEngine.UI.RawImage>().texture = frameAwaiter.Texture;
            }
        }
    }
}
```

## Character System

### Expression Support

Characters support 7 built-in expressions, each with its own index:

- **0**: talk-neutral (default speaking)
- **1**: approve (nodding, positive)
- **2**: disapprove (negative reaction)
- **3**: smile (happy expression)
- **4**: sad (sorrowful expression)  
- **5**: surprised (shocked reaction)
- **6**: confused (puzzled expression)

Use `expressionIndex: -1` in `SpeakAsync()` to generate voice-only output without video frames.

### Character Formats

Characters support two storage formats:

#### Bundle Format (.bundle) - macOS
- Character data stored in a `.bundle` directory
- Appears as a single file in macOS Finder
- Contains `Info.plist` for proper macOS package metadata
- Automatically used on macOS platforms

#### Folder Format - Universal  
- Character data stored in a regular directory
- Works on all platforms (Windows, macOS, Linux)
- Used on non-macOS platforms or when explicitly requested

### Character Data Structure

Each character contains:
- **character.json**: Character configuration (name, gender, pitch, speed, intro)
- **image.png**: Character portrait image
- **drivingFrames/**: Expression data for each expression index
  - **expression-N/**: Folder for expression N
    - **XXXXX.png**: Generated driving frames
    - **latents.bin**: Precomputed latent representations
    - **faces.json**: Face detection and processing data
    - **textures/**: Precomputed texture data
- **voice/**: Voice model and configuration
  - **sample.wav**: Reference voice sample
  - **voice_config.json**: Voice generation parameters

## API Reference

### LiveTalkAPI (Singleton)

#### Initialization
```csharp
LiveTalkAPI.Instance.Initialize(
    LogLevel logLevel = LogLevel.INFO,
    string characterSaveLocation = "",
    string parentModelPath = "",
    MemoryUsage memoryUsage = MemoryUsage.Balanced
)
```

#### Character Management
```csharp
// Create character with full options
IEnumerator CreateCharacterAsync(
    string name, 
    Gender gender, 
    Texture2D image, 
    Pitch pitch, 
    Speed speed, 
    string intro,
    string voicePromptPath,
    Action<Character> onComplete, 
    Action<Exception> onError,
    CreationMode creationMode = CreationMode.AllExpressions,
    bool useBundle = true
)

// Load character by ID (from save location)
IEnumerator LoadCharacterAsyncFromId(
    string characterId, 
    Action<Character> onComplete, 
    Action<Exception> onError
)

// Load character from specific path
IEnumerator LoadCharacterAsyncFromPath(
    string characterPath, 
    Action<Character> onComplete, 
    Action<Exception> onError
)

// Get available characters
string[] GetAvailableCharacterIds()
string CharacterSaveLocation { get; }

// Check if bundle format is supported on current platform
static bool CanUseBundle()
```

#### Animation Generation
```csharp
// LivePortrait animation from texture list
FrameStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, List<Texture2D> drivingFrames)

// LivePortrait animation from video
FrameStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, VideoPlayer videoPlayer, int maxFrames = -1)

// LivePortrait animation from directory
FrameStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, string drivingFramesPath, int maxFrames = -1)

// MuseTalk lip sync
FrameStream GenerateTalkingHeadAsync(Texture2D avatarTexture, string talkingHeadFolderPath, AudioClip audioClip)
```

### Character Class

#### Properties
```csharp
string Name { get; }
string CharacterId { get; }
Gender Gender { get; }
Texture2D Image { get; }
Pitch Pitch { get; }
Speed Speed { get; }
string Intro { get; }
bool IsDataLoaded { get; }
```

#### Methods
```csharp
// Make character speak with animation (pipelined audio/animation)
IEnumerator StartSpeakWithCallbacks(
    string text, 
    int expressionIndex = 0,  // Use -1 for voice-only
    Action<FrameStream, AudioClip> onAudioReady = null,  // Called when audio is ready
    Action onAnimationComplete = null,  // Called when all frames are generated
    Action<Exception> onError = null
)

// Legacy method (still supported but StartSpeakWithCallbacks is preferred)
IEnumerator SpeakAsync(
    string text, 
    int expressionIndex = 0,  // Use -1 for voice-only
    Action<FrameStream, AudioClip> onComplete = null, 
    Action<Exception> onError = null
)

// Static methods for loading
static IEnumerator LoadCharacterAsyncFromPath(
    string characterPath,
    Action<Character> onComplete,
    Action<Exception> onError
)

static IEnumerator LoadCharacterAsyncFromId(
    string characterId,
    Action<Character> onComplete,
    Action<Exception> onError
)
```

### CharacterPlayer Component

A reusable MonoBehaviour that handles character loading, idle animation, speech playback, and smooth transitions. This is the recommended way to integrate LiveTalk characters into your Unity scenes.

#### Features

- **Auto-loads character** when assigned
- **Idle animation** playback (expression 0) at 25 FPS with ping-pong cycling
- **Speech queueing** with automatic playback
- **Smooth transitions** between idle and speech states
- **Event-driven** architecture for UI integration
- **Audio-only mode** for characters without avatars (narrators, phone voices, etc.)
- **Static image fallback** for characters without idle animations

#### Quick Setup

```csharp
using LiveTalk.API;
using UnityEngine;
using UnityEngine.UI;

public class CharacterPlayerExample : MonoBehaviour
{
    [SerializeField] private Character myCharacter;
    
    void Start()
    {
        // CharacterPlayer is automatically created by the Character
        var player = myCharacter.CharacterPlayer;
        
        // Subscribe to events
        player.OnFrameUpdate += (frame) => {
            // Display frame in your UI (e.g., RawImage)
            GetComponent<RawImage>().texture = frame;
        };
        
        player.OnSpeechStarted += () => Debug.Log("Character started speaking");
        player.OnSpeechEnded += () => Debug.Log("Character finished speaking");
        player.OnCharacterLoaded += () => Debug.Log("Character loaded and ready");
        
        // Queue speech (automatically plays idle animation between speeches)
        player.QueueSpeech("Hello! I'm ready to talk.", expressionIndex: 0);
        player.QueueSpeech("This is my second line.", expressionIndex: 3); // smile
    }
}
```

#### Properties

```csharp
PlaybackState State { get; }           // Uninitialized, Loading, Idle, Speaking, Paused
Character Character { get; }           // The LiveTalk Character
Texture DisplayImage { get; }          // Current display frame
bool HasQueuedSpeech { get; }          // True if speech is in queue
int QueuedSpeechCount { get; }         // Number of queued speech items
```

#### Methods

```csharp
// Queue a single speech line
void QueueSpeech(string text, int expressionIndex = 0, bool withAnimation = true)

// Queue multiple speech lines at once
void QueueSpeechBatch(List<string> textLines, int expressionIndex = 0, bool withAnimation = true)

// Control playback
void Pause()
void Resume()
void Stop()              // Stops all speech and returns to idle
void ClearQueue()        // Removes queued speech
```

#### Events

```csharp
event Action<Texture> OnFrameUpdate;      // Fired for each new frame (idle or speech)
event Action OnSpeechStarted;             // Speech playback started
event Action OnSpeechEnded;               // Speech playback ended
event Action<Exception> OnError;          // Error occurred
event Action OnCharacterLoaded;           // Character finished loading
event Action OnIdleStarted;               // Idle animation started
```

#### Audio-Only Mode

For characters without avatars (narrators, phone voices, etc.):

```csharp
// CharacterPlayer automatically detects if character has no idle frames
// and switches to audio-only mode
player.QueueSpeech("This is narrator voice without animation.", withAnimation: false);
```

#### Integration with UI Toolkit

CharacterPlayer works seamlessly with Unity UI Toolkit by firing `OnFrameUpdate` events that you can display in VisualElements:

```csharp
// In your UI Controller
private VisualElement _avatarDisplay;
private CharacterPlayer _player;

void SetupCharacter(Character character)
{
    _player = character.CharacterPlayer;
    
    // Display frames in UI Toolkit VisualElement
    _player.OnFrameUpdate += (frame) => {
        if (_avatarDisplay != null)
        {
            _avatarDisplay.style.backgroundImage = new StyleBackground(frame);
        }
    };
}
```

### DialogueOrchestrator Component

Orchestrates multi-character turn-based dialogue, handling speaker switching, audio coordination, and visual display management. Perfect for conversations, cutscenes, and interactive dialogues.

#### Features

- **Multi-character support** with automatic speaker switching
- **Turn-based dialogue** with automatic queuing
- **Visual switching** - shows speaking character's animation
- **Audio coordination** - ensures only one character speaks at a time
- **Event-driven** - integrates easily with UI systems
- **Supports audio-only characters** (narrators)

#### Quick Setup

```csharp
using LiveTalk.API;
using UnityEngine;
using UnityEngine.UI;

public class DialogueExample : MonoBehaviour
{
    [SerializeField] private Character detective;
    [SerializeField] private Character suspect;
    [SerializeField] private RawImage dialogueDisplay;
    
    private DialogueOrchestrator _orchestrator;
    
    void Start()
    {
        // Create orchestrator
        var orchestratorObj = new GameObject("DialogueOrchestrator");
        _orchestrator = orchestratorObj.AddComponent<DialogueOrchestrator>();
        
        // Register characters
        _orchestrator.RegisterCharacter("detective", detective.CharacterPlayer);
        _orchestrator.RegisterCharacter("suspect", suspect.CharacterPlayer);
        
        // Subscribe to events
        _orchestrator.OnFrameUpdate += (frame) => {
            dialogueDisplay.texture = frame;
        };
        
        _orchestrator.OnSpeakerChanged += (speakerId) => {
            Debug.Log($"Now speaking: {speakerId}");
        };
        
        // Queue a conversation
        PlayConversation();
    }
    
    void PlayConversation()
    {
        _orchestrator.QueueDialogue("detective", "Where were you on the night of the murder?", expressionIndex: 0);
        _orchestrator.QueueDialogue("suspect", "I was at home, I swear!", expressionIndex: 5); // surprised
        _orchestrator.QueueDialogue("detective", "Can anyone confirm that?", expressionIndex: 0);
        _orchestrator.QueueDialogue("suspect", "My... my wife can.", expressionIndex: 4); // sad
    }
}
```

#### Batch Dialogue

Queue multiple dialogue segments at once:

```csharp
var conversation = new List<DialogueOrchestrator.DialogueSegment>
{
    new() { CharacterId = "detective", Text = "Let's review the evidence.", ExpressionIndex = 0 },
    new() { CharacterId = "suspect", Text = "I didn't do it!", ExpressionIndex = 5 },
    new() { CharacterId = "detective", Text = "Then explain this.", ExpressionIndex = 0 }
};

_orchestrator.QueueDialogueBatch(conversation);
```

#### Properties

```csharp
bool IsPlaying { get; }                 // True if dialogue is currently playing
int QueuedDialogueCount { get; }        // Number of queued dialogue segments
string CurrentSpeakerId { get; }        // ID of currently speaking character
```

#### Methods

```csharp
// Register/unregister characters
void RegisterCharacter(string characterId, CharacterPlayer player)
void UnregisterCharacter(string characterId)

// Queue dialogue
void QueueDialogue(string characterId, string text, int expressionIndex = 0, bool withAnimation = true)
void QueueDialogueBatch(List<DialogueSegment> segments)

// Control
void Stop()              // Stop all dialogue
void ClearQueue()        // Remove queued dialogue
```

#### Events

```csharp
event Action<string> OnSpeakerChanged;    // Fired when active speaker changes (characterId)
event Action<Texture> OnFrameUpdate;      // Fired for each frame of current speaker
event Action OnDialogueStarted;           // Dialogue sequence started
event Action OnDialogueEnded;             // All dialogue completed
event Action<Exception> OnError;          // Error occurred
```

#### With Narrator (Audio-Only)

```csharp
// Register narrator as audio-only character
_orchestrator.RegisterCharacter("narrator", narratorCharacter.CharacterPlayer);

// Queue narrator dialogue without animation
_orchestrator.QueueDialogue("narrator", "Meanwhile, in another part of town...", 
    expressionIndex: 0, withAnimation: false);
```

#### Integration with UI Toolkit

```csharp
// In your UI Controller
private DialogueOrchestrator _orchestrator;
private VisualElement _speakerDisplay;
private Label _speakerNameLabel;

void SetupDialogue()
{
    // Subscribe to frame updates
    _orchestrator.OnFrameUpdate += (frame) => {
        _speakerDisplay.style.backgroundImage = new StyleBackground(frame);
    };
    
    // Update UI when speaker changes
    _orchestrator.OnSpeakerChanged += (speakerId) => {
        _speakerNameLabel.text = speakerId;
    };
    
    _orchestrator.OnDialogueEnded += () => {
        Debug.Log("Conversation finished!");
    };
}
```

### FrameStream Class

#### Properties
```csharp
int TotalExpectedFrames { get; set; }
bool HasMoreFrames { get; }
```

#### Methods
```csharp
FrameAwaiter WaitForNext() // For use in coroutines
bool TryGetNext(out Texture2D texture) // Non-blocking retrieval
```

## Configuration Options

### LogLevel Enum
- `VERBOSE`: Detailed debugging information
- `INFO`: General information messages (default)
- `WARNING`: Warning messages only
- `ERROR`: Error messages only

### MemoryUsage Enum
- `Quality`: Uses FP32 models for maximum quality (not recommended - minimal quality improvement)
- `Performance`: Loads all models upfront for faster first-time inference (desktop)
- `Balanced`: Loads models on-demand, recommended for desktop devices (default)
- `Optimal`: Minimal memory footprint, recommended for mobile devices

### CreationMode Enum
- `VoiceOnly`: Only generates voice, no visual expressions
- `SingleExpression`: Generates voice and talk-neutral expression only
- `AllExpressions`: Generates voice and all 7 expressions (default)

### Character Configuration
- **Gender**: `Male`, `Female`
- **Pitch**: `VeryLow`, `Low`, `Moderate`, `High`, `VeryHigh`
- **Speed**: `VeryLow`, `Low`, `Moderate`, `High`, `VeryHigh`

## Requirements

- Unity 6000.0.46f1 or later
- Platforms: macOS (CPU/CoreML), Windows (Not tested)
- Minimum 32GB RAM recommended for character creations
- Storage space for models (~10GB total: ~7GB LiveTalk + ~3GB SparkTTS)

## Performance

**MacBook Pro M4 Max (Onnx with CoreML Execution Provider)**:
  - Speech With LipSync generation - 10-11 FPS
  - Character Creation - 10 minutes per character (all expressions)
  - Character Creation - ~2 minutes per character (single expression)
  
### Model Execution Times (Mac M4)

**LivePortrait Pipeline - 4 FPS**:
- `motion_extractor` (FP32): 30-60ms
- `warping_spade` (FP16): 180-250ms  
- `landmark_runner` (FP32): 2-3ms

**MuseTalk Pipeline - 11-12 FPS**:
- `vae_encoder` (FP16): 20-30ms
- `unet` (FP16): 30-40ms
- `vae_decoder` (FP16): 30-50ms

## License

This project is licensed under the MIT License, following the licensing of the underlying technologies:

- **LivePortrait**: Licensed under the MIT License
- **MuseTalk**: Licensed under the MIT License  
- **SparkTTS**: Licensed under the Apache License 2.0
- **Other dependencies**: Licensed under their respective open-source licenses

See the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project incorporates code and models from several open-source projects:
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) - Portrait animation technology
- [MuseTalk](https://github.com/TMElyralab/MuseTalk) - Real-time lip synchronization  
- [SparkTTS](https://github.com/arghyasur1991/Spark-TTS-Unity) - Text-to-speech synthesis
- ONNX Runtime - Cross-platform ML inference

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## Credits

- **LivePortrait Team** at KwaiVGI for portrait animation technology
- **MuseTalk Team** at TMElyralab for lip synchronization technology
- **SparkTTS Team** for text-to-speech synthesis
- ONNX Runtime team for cross-platform ML inference

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.
