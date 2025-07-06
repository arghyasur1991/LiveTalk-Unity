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
- com.github.asus4.onnxruntime (0.4.0)
- com.github.asus4.onnxruntime-extensions (0.4.0)
- com.unity.nuget.newtonsoft-json (3.2.1)

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

* **Smart Model Selection**: Automatically determines which models are actually used by analyzing the LiveTalk codebase
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

The tool automatically selects the optimal precision for each model based on the LiveTalk codebase:

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
3. Copy the extracted `LiveTalk` folder with models to your Unity project's `Assets/StreamingAssets/` directory

#### SparkTTS Models
Download the pre-exported ONNX models from [Google Drive](https://drive.google.com/file/d/1YXj81ApcEasY17a8Zj9RqTpvn4s1UKk7/view?usp=sharing).

1. Download the ZIP file from the link
2. Extract the contents  
3. Copy the extracted `SparkTTS` folder with models to your Unity project's `Assets/StreamingAssets/` directory

### Exporting Models (Coming Soon)

Coming Soon - conversion scripts to export models from the original Python repositories:

- **LivePortrait**: [https://github.com/KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- **MuseTalk**: [https://github.com/TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)

The export scripts will convert PyTorch models to ONNX format and apply CoreML optimizations for Unity integration.

## Usage

### Basic Setup

```csharp
using UnityEngine;
using LiveTalk.API;
using System.Collections;

public class LiveTalkExample : MonoBehaviour
{
    void Start()
    {
        // Initialize the LiveTalk system
        LiveTalkAPI.Instance.Initialize(
            logLevel: LogLevel.INFO,
            initializeModelsOnDemand: true, // Load models when needed (default: true)
            characterSaveLocation: "", // Uses default location
            parentModelPath: "" // Uses StreamingAssets
        );
    }
}
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
        
        // Create a new character
        yield return LiveTalkAPI.Instance.CreateCharacterAsync(
            name: "MyCharacter",
            gender: Gender.Female,
            image: characterImage,
            pitch: Pitch.Moderate,
            speed: Speed.Moderate,
            intro: "Hello, I am your virtual assistant!",
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
        
        // Load an existing character
        string characterId = "your-character-id";
        yield return LiveTalkAPI.Instance.LoadCharacterAsync(
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
        
        yield return loadedCharacter.SpeakAsync(
            text: "Hello! I can speak with realistic lip sync!",
            expressionIndex: 0, // Use talk-neutral expression
            onComplete: (frameStream, audioClip) => {
                // Process the generated frames and audio
                StartCoroutine(PlayGeneratedVideo(frameStream, audioClip));
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
        
        // Process video frames
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
    bool initializeModelsOnDemand = true,
    string characterSaveLocation = "",
    string parentModelPath = ""
)
```

#### Character Management
```csharp
// Create character
IEnumerator CreateCharacterAsync(string name, Gender gender, Texture2D image, 
    Pitch pitch, Speed speed, string intro, Action<Character> onComplete, Action<Exception> onError)

// Load character
IEnumerator LoadCharacterAsync(string characterId, Action<Character> onComplete, Action<Exception> onError)

// Get available characters
string[] GetAvailableCharacterIds()
string GetCharacterPath(string characterId)
string GetCharacterFormat(string characterId)
bool IsCharacterBundle(string characterId)
bool IsCharacterFolder(string characterId)
```

#### Animation Generation
```csharp
// LivePortrait animation
FrameStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, List<Texture2D> drivingFrames)
FrameStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, VideoPlayer videoPlayer, int maxFrames = -1)
FrameStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, string drivingFramesPath, int maxFrames = -1)

// MuseTalk lip sync
FrameStream GenerateTalkingHeadAsync(Texture2D avatarTexture, string talkingHeadFolderPath, AudioClip audioClip)
```

### Character Class

#### Properties
```csharp
string Name { get; }
Gender Gender { get; }
Texture2D Image { get; }
Pitch Pitch { get; }
Speed Speed { get; }
string Intro { get; }
bool IsDataLoaded { get; }
```

#### Methods
```csharp
// Create character avatar data
IEnumerator CreateAvatarAsync()
IEnumerator CreateAvatarAsync(bool useBundle)

// Make character speak
IEnumerator SpeakAsync(string text, int expressionIndex = 0, 
    Action<FrameStream, AudioClip> onComplete = null, Action<Exception> onError = null)
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
- `INFO`: General information messages
- `WARNING`: Warning messages only
- `ERROR`: Error messages only

### Initialization Options
- **initializeModelsOnDemand**: When `true` (default), models are loaded only when needed for inference, reducing startup time and memory usage. When `false`, all models are loaded immediately during initialization for faster first-time inference.

### Character Configuration
- **Gender**: `Male`, `Female`
- **Pitch**: `VeryLow`, `Low`, `Moderate`, `High`, `VeryHigh`
- **Speed**: `VeryLow`, `Low`, `Moderate`, `High`, `VeryHigh`

## Requirements

- Unity 6000.0 or later
- Platforms: macOS (CPU/CoreML), Windows (Not tested)
- Minimum 32GB RAM recommended for character creations
- Storage space for models (~6GB total: ~7GB LiveTalk + ~3GB SparkTTS)

## Performance

**MacBook Pro M4 Max (Onnx with CoreML Execution Provider)**:
  - Speech With LipSync generation - 10-11 FPS
  - Character Creation - 10 minutes per character
  
### Model Execution Times (Mac M4)

**LivePortrait Pipeline - 4 FPS**:
- `motion_extractor` (FP32): 30-60ms
- `warping_spade` (FP16): 180-250ms  
- `landmark_runner` (FP32): 2-3ms

**MuseTalk Pipeline - 11 FPS**:
- `vae_encoder` (FP16): 20-30ms
- `unet` (FP16): 30-40ms
- `vae_decoder` (FP16): 40-50ms

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