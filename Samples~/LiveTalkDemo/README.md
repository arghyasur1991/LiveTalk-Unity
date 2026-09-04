# LiveTalk Demo

One MonoBehaviour, `LiveTalkDemo`, that exercises the whole 2.0 flow from a
handful of uGUI controls. It uses only `UnityEngine.UI` (`Text`, `Button`,
`RawImage`, `Slider`), which LiveTalk already depends on, so it compiles with
nothing extra installed.

## Set up

1. Import the sample from **Window → Package Manager → LiveTalk Unity →
   Samples**.
2. Deploy the LivePortrait / MuseTalk models and point the TTS package at its
   weights — see the package README.
3. In a scene, add `LiveTalkDemo` to a GameObject and wire the inspector:
   - **Source Image**: a readable portrait `Texture2D`.
   - **Audio Clip** (optional): for the *Generate Talking Head* button.
   - **Audio Source**: plays generated speech.
   - **Video Player** (optional): driving frames for *Generate Animated*;
     otherwise the `Driving Frames Folder Path` under `StreamingAssets` is used.
   - **UI**: five `Button`s, two `Text`s (status, FPS), a `RawImage` preview
     and a `Slider` for progress.

## Buttons

| Button | Calls |
|---|---|
| Generate Animated | `LiveTalkAPI.GenerateAnimatedTexturesAsync` (LivePortrait only) |
| Generate Talking Head | `LiveTalkAPI.GenerateTalkingHeadAsync` (MuseTalk on a static portrait + clip) |
| Create Character | `CreateAvatarAsync` → `DesignVoiceAsync` → `CreateCharacter`; prints the ids and auditions `Voice.Sample` |
| Load Character | `LoadCharacterAsyncFromId(characterIdToLoad)` |
| Speak | `Character.SpeakAsync` with expression 0, or `-1` when *Generate Voice Only* is ticked |

The first *Create Character* on a portrait takes minutes (the avatar is built);
every later one on the same portrait reuses it and only the voice is new.
