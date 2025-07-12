# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
