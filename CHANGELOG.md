# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Driving frames are cached on the source image, so a second character built from the same portrait reuses the previous avatar pass instead of spending minutes in LivePortrait again. Frames are a pure function of the image and the requested expression set, and the cache key covers both — a single-expression folder will not satisfy a request for the full set. The entry is only written once every expression has finished, so an aborted run cannot leave a short folder for a later character to adopt. This matters most when only the *voice* changes: re-rolling a designed voice or locking a clone previously paid for the whole avatar bake again. Requires the cache to be enabled.
- `EnsureRuntimeHost()` recreates the coroutine GameObject after Play teardown while the API singleton is still initialized.
- `UnloadTts()` drops the TTS engine's ONNX sessions without disposing LivePortrait / MuseTalk.
- `VoiceModelsLoaded` reports whether the TTS engine currently holds models.
- `voiceInstruct` on character creation, passed through to voice design. `Gender`/`Pitch`/`Speed` are composed into a natural-language description in `Utils/VoiceInstruct`, which is host policy rather than engine behaviour.

### Changed
- **TTS backend is now Qwen3-TTS** (`com.genesis.qwentts.unity`) instead of Spark-TTS. Voice design takes a description; cloning takes a reference recording plus its transcript.
- `ModelUtils` uses the default `OrtEnv` and forwards log attribution to whichever library created the environment. ONNX Runtime allows one environment per process, so creating a second one with its own sink meant LiveTalk's model names never reached the sink that was actually installed.
- `SpeakAsync` takes an optional `onSpeechChunk` callback and streams speech as it is generated instead of only when the line finishes: the first chunk arrives in about a second. Delivered on the main thread. Ignored on a cache hit, where the whole clip is already on disk.
- Audio helpers delegate to the TTS package's `QwenAudio` rather than carrying their own copies. `ConcatenateAudioClips` now honours its `silenceDuration` argument, which the previous delegating version ignored, and WAV load/save is local in `AudioFileIO`.
- `Character.CreateAvatarAsync` takes an optional trailing `onError`. `FrameStream` exposes `Error`, set when the producer that fills it failed; the stream is still marked finished so consumers drain and exit.

### Fixed
- Faulted tasks inside coroutines are no longer skipped. The pipeline waited on `task.IsCompleted`, which is also true for a faulted task, so a failed model load, synthesis or file read was silently passed over and surfaced later as a misleading error ("Model is not initialized", "Character voice not loaded", "Generated audio clip is null") — or not at all. Every such wait now goes through an internal bridge (`TaskYield.Wait`) that logs the original exception with its stack and rethrows it inside the coroutine.
- Frame producers (`LivePortraitInference`, `MuseTalkInference`, `LiveTalkController`) mark their `FrameStream` finished in `finally`, on every exit path. Previously a fault mid-generation left the stream open, the consumer waiting forever, and — for lip-sync — the MuseTalk lease never released, so the next animated line blocked on acquire with no error anywhere. A driving-frame or lip-sync producer that fails is now reported as a failure by its consumer rather than as a shorter clip, and a partially written frames-cache entry is deleted instead of being taken as a hit next time.
- `VoiceQueue` / `MuseTalkQueue` leases are acquired through the bridge and released in `finally`, so a fault or a stopped coroutine cannot leak the lock.
- `CreateCharacterAsync`, `CreateAvatarAsync`, `LoadCharacterAsync*` and `SpeakAsync` honour their `onError` contract: exactly one of `onComplete` / `onError` fires, and `onComplete` is never called with a half-built character. Voice design, voice clone and voice load throw with the offending path instead of logging and returning, and a character is not marked loaded when its voice is missing or failed to load. A failed animation still fires `SpeakAsync`'s `onError` after `onAudioReady` and never fires `onAnimationComplete`.
- Error messages no longer refer to a `CharacterFactory` that does not exist; the downstream messages that remain carry a hint to look for the earlier, original error.

## [1.0.0] - 2025-07-04

### Added
- **LiveTalk-Unity Package**: Complete Unity package for real-time talking head generation
- **Dual-Pipeline System**: LivePortrait for facial animation + MuseTalk for lip synchronization
- **Character Creation System**: Create, save, and load characters with multiple expressions and voices
- **Advanced Character Management**: Support for both folder and macOS bundle formats
- **Complete API**: LiveTalkAPI singleton and Character class with full documentation
- **Expression Support**: 7 built-in expressions (talk-neutral, approve, disapprove, smile, sad, surprised, confused)
- **Integrated TTS**: Built-in SparkTTS integration for voice generation
- **Model Download Links**: Pre-exported ONNX models for both LiveTalk and SparkTTS
- **Cross-Platform Support**: macOS (CPU/CoreML), Windows (Not tested)
- **Performance Optimizations**: CoreML optimization for efficient on-device inference

### Features
- **Real-time Animation**: Generate talking head videos from avatar images and audio
- **Character Persistence**: Save and load character data with precomputed expressions
- **Voice Synthesis**: Create character voices with configurable pitch, speed, and gender
- **Frame Streaming**: Efficient frame-by-frame processing with coroutine support
- **Multiple Input Formats**: Support for images, videos, and directory-based driving frames
- **Bundle Support**: macOS package format for seamless character distribution
- **Memory Management**: Optimized memory usage with unsafe code and parallel processing

### Performance
- **Overall Performance**: 10-11 FPS for speech with lip sync on Mac M4 Max
- **Character Creation**: 10 minutes per character on Mac M4 Max
- **LivePortrait Pipeline**: 4 FPS
  - motion_extractor (FP32): 30-60ms
  - warping_spade (FP16): 180-250ms
  - landmark_runner (FP32): 2-3ms
- **MuseTalk Pipeline**: 11 FPS
  - vae_encoder (FP16): 20-30ms
  - unet (FP16): 30-40ms
  - vae_decoder (FP16): 40-50ms

### Requirements
- Unity 6000.0 or later
- Minimum 32GB RAM for character creation
- Storage space: ~10GB total (~7GB LiveTalk + ~3GB SparkTTS)
- macOS (CPU/CoreML) or Windows (Not tested)

### Dependencies
- com.github.asus4.onnxruntime (0.4.0)
- com.github.asus4.onnxruntime-extensions (0.4.0)
- com.unity.nuget.newtonsoft-json (3.2.1)

### License
- MIT License (following LivePortrait and MuseTalk licensing)
- Apache License 2.0 for SparkTTS components
