# Clipz - Turn long videos into viral clips automatically.

![Viral Clip Extractor](doc/img.png)

<p align="center">
  <b>AI-powered instruction-driven multimodal video clip extraction</b><br/>
  Audio • Visual • Speech • LLM Reasoning
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/LLM-Powered-purple" alt="LLM Powered"/>
  <img src="https://img.shields.io/badge/Multimodal-Audio%20%7C%20Video%20%7C%20Text-orange" alt="Multimodal"/>
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Active"/>
</p>

<p align="center">
  <a href="#quick-start">⚡ Quick Start</a> •
  <a href="#-why-clipz">🤔 Why Clipz?</a> •
  <a href="#features">✨ Features</a> •
  <a href="#api-reference">📚 API</a> •
  <a href="#future-roadmap">🚀 Roadmap</a>
</p>

---

## 🤔 Why Clipz?

Many existing clip tools already do a great job with:
- audio-based excitement detection  
- visual motion & scene analysis  
- even basic LLM-assisted highlight detection  

**Clipz goes one step further — it’s instruction-driven.**

Instead of passively finding “hot moments,” you tell the system *what you want*:
- *“Extract the funniest moments”*
- *“Only clip Speaker B”*
- *“Find emotionally intense reactions”*

An LLM interprets your intent and grounds it using:
✔ audio cues & prosody  
✔ visual signals & scene context  
✔ sentence-aware transcription  

So clips aren’t just *popular* — they’re **exactly aligned with your instruction**.

---


## Quick Start

👉 **New to this project?** Check out the [Quick Start Guide](doc/QUICKSTART.md) for installation and first run in 5 minutes!

## Features

### Core Capabilities

- 🎵 **Multi-Scale Audio Analysis**: Detects excitement through loudness, spectral novelty, rhythm, prosody, and semantic events (laughter, applause)
- 🎬 **Advanced Visual Analysis**: Tracks motion, semantic surprise (CLIP), composition quality, shot boundaries, and face detection
- 🗣️ **Speech Transcription**: Word-level timestamps using Whisper, respects sentence boundaries for natural clips
- 🤖 **LLM-Powered Intelligence**: Instruction-driven clip extraction with semantic merging and context-aware ranking
- ⚡ **Parallel Processing**: Multi-threaded feature extraction with automatic caching
- 🎯 **Smart Clip Boundaries**: Never cuts mid-sentence, aligns to natural speech segments

## API Reference

### Main Pipeline (`main.py`)

```python
from main import ViralClipExtractor

# Initialize with custom weights
extractor = ViralClipExtractor(
    audio_weight=0.5,      # 0-1, weight for audio excitement
    video_weight=0.5,      # 0-1, weight for visual excitement
    use_cache=True,        # Cache features for faster re-runs
    output_dir="output"    # Output directory
)

# Process video end-to-end
results = extractor.process(
    video_path="video.mp4",
    user_query="give me 10 interesting clips",  # Natural language query
    target_fps=2,          # Video analysis FPS (lower=faster)
    min_duration=5,        # Minimum clip length (seconds)
    max_duration=60,       # Maximum clip length (seconds)
    export=True            # Export video files
)

# Access results
for clip in results["clips"]:
    print(f"Time: {clip['start']:.1f}s - {clip['end']:.1f}s")
    print(f"Transcript: {clip['transcript']}")
    print(f"Score: {clip['llm_interest_score']}/10")
    print(f"Reason: {clip['reason']}")
    print(f"Tags: {clip['tags']}")
```

### Individual Modules

#### Audio Analysis (`Audio/audio.py`)

**Features Extracted:**
- Multi-scale loudness (short/long-term RMS)
- Spectral novelty via MFCC delta
- Rhythm variance and onset strength
- Silence contrast and dramatic pauses
- Structural boundaries (change-point detection)
- Semantic events (laughter, applause, cheering) via YAMNet

```python
from Audio.audio import ClipAudio

detector = ClipAudio(sr=16000)  # 16kHz optimized for speed
timestamps, scores = detector.compute_audio_scores(
    audio_path="audio.wav",
    use_cache=True
)
```

#### Video Analysis (`video/video.py`)

**Features Extracted:**
- Optical flow motion magnitude
- CLIP semantic surprise detection
- Composition scoring (rule of thirds)
- Shot boundary detection
- Face detection and tracking
- Temporal rhythm analysis

```python
from video.video import ClipVideo

detector = ClipVideo()
timestamps, scores = detector.compute_visual_scores(
    video_path="video.mp4",
    target_fps=2,
    use_cache=True
)
```

#### Transcription (`Transcription/transcribe.py`)

**Returns:** List of sentence segments with timestamps

```python
from Transcription.transcribe import Transcriber

segments = Transcriber.transcribe_with_timestamps(
    audio_path="audio.wav",
    model_size="base",  # tiny/base/small/medium/large
    verbose=False
)
# [{"start": 0.0, "end": 3.5, "text": "Hello world"}, ...]
```

#### LLM Integration (`LLM/llm.py`)

**Uses:** OpenRouter API for GPT-4o-mini

```python
from LLM.llm import LLM

llm = LLM()
response = llm.generate_text(
    prompt="Your prompt here",
    model="openai/gpt-4o-mini",
    max_tokens=2000,
    temperature=0.3
)
```

## Command Line Options

```
usage: main.py [-h] [--query QUERY] [--audio-weight AUDIO_WEIGHT]
               [--video-weight VIDEO_WEIGHT] [--fps FPS]
               [--min-duration MIN_DURATION] [--max-duration MAX_DURATION]
               [--output-dir OUTPUT_DIR] [--no-export]
               video_path

positional arguments:
  video_path            Path to input video file

optional arguments:
  -h, --help            show this help message and exit
  --query QUERY         Query for clip selection (default: "give me 10 interesting clips")
  --audio-weight AUDIO_WEIGHT
                        Weight for audio scores 0-1 (default: 0.5)
  --video-weight VIDEO_WEIGHT
                        Weight for video scores 0-1 (default: 0.5)
  --fps FPS             Target FPS for video analysis (default: 2)
  --min-duration MIN_DURATION
                        Minimum clip duration in seconds (default: 5)
  --max-duration MAX_DURATION
                        Maximum clip duration in seconds (default: 60)
  --output-dir OUTPUT_DIR
                        Output directory for clips (default: "output")
  --no-export           Skip exporting video files
```

## Output

The system generates:

### Video Clips
- **Location**: `output/clips_<timestamp>/`
- **Format**: Individual MP4 files (`clip_001.mp4`, `clip_002.mp4`, etc.)
- **Content**: Extracted video segments ready to use

### Metadata & Cache
- **Clip Metadata**: `.cache/metadata/` - Individual JSON files for each clip
- **Analysis Report**: `.cache/analysis/` - Complete analysis metadata
- **Feature Cache**: `.cache/audio/` and `.cache/video/` - Cached features for faster re-runs
- **Transcription Cache**: `.cache/transcription/` - Cached transcripts

Example clip metadata:
```json
{
  "clip_number": 1,
  "video_file": "output/clips_20260110_192101/clip_001.mp4",
  "start_time": 45.2,
  "end_time": 58.7,
  "duration": 13.5,
  "transcript": "...",
  "interest_score": 9.5,
  "reason": "Emotional storytelling with dramatic pause",
  "tags": ["emotional", "dramatic"]
}
```

**Clean Output**: Your `output/` folder only contains the video clips - all metadata and cache files are organized in `.cache/` to keep things tidy!

## Advanced Usage

### Individual Module Testing

Test audio analysis:
```bash
python audio.py path/to/audio.wav
```

Test video analysis:
```bash
python video.py path/to/video.mp4
```

Test transcription:
```bash
python transcribe.py path/to/audio.wav
```

### Caching

The system automatically caches expensive computations in the `.cache/` directory:
- **Audio features**: `.cache/audio/audio_cache_<hash>.npz`
- **Video features**: `.cache/video/visual_cache_<hash>.npz`
- **Transcriptions**: `.cache/transcription/transcript_<hash>.json`
- **Metadata**: `.cache/metadata/` and `.cache/analysis/`

This makes subsequent runs much faster! To disable caching:
```python
extractor = ClipExtractor(use_cache=False)
```

## Performance Tips

- **GPU Acceleration**: Install CUDA-enabled PyTorch for faster processing
- **Lower FPS**: Use `--fps 1` for faster video analysis (less accurate)
- **Smaller Models**: Whisper uses "base" model by default (good balance)
- **Cache Results**: Re-runs on the same video are much faster with caching

## Dependencies

Key dependencies:
- **ultralytics**: YOLOv8 object detection
- **transformers**: CLIP model for semantic analysis
- **whisper**: Speech transcription
- **librosa**: Audio analysis
- **opencv-python**: Video processing
- **torch**: Deep learning framework
- **dlib**: Face detection
- **praat-parselmouth**: Prosody analysis

See `requirements.txt` for complete list.

## Troubleshooting

### YOLO Model Download

The YOLOv8 model (`yolov8n.pt`) is **automatically downloaded** on first run by the Ultralytics package. You don't need to manually download it.

If you encounter issues:
- Ensure you have internet connection on first run
- The model (~6MB) downloads to Ultralytics cache
- Check firewall settings if download fails

### FFmpeg Not Found

Install FFmpeg:
- **Windows**: Download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

### OpenRouter API Errors

Check your `.env` file has a valid `OPENROUTER_API_KEY`.

### Out of Memory

Try:
- Lowering `target_fps` (default is 2)
- Processing shorter videos
- Closing other applications

## Known Limitations

### Multi-Language Videos
Videos containing multiple languages may produce unexpected clips because Whisper translates everything into English by default. This can result in:
- Loss of context from non-English speech
- Incorrect clip boundaries due to translation timing differences
- Mixed language content being merged incorrectly

**Workaround**: For better results with multi-language content, process each language segment separately or use the `task="transcribe"` parameter to keep original language.

### Long Video Processing
Videos longer than 1 hour may consume significant processing time (30-60+ minutes depending on hardware):
- Audio feature extraction scales with video duration
- Video analysis requires processing thousands of frames
- LLM analysis has context window limits for very long videos

**Tips for long videos**:
- Use lower `target_fps` (1 instead of 2) for faster processing
- Enable caching to avoid re-processing if you need to re-run
- Consider splitting very long videos into smaller segments
- Ensure sufficient RAM (16GB+ recommended for 1-hour videos)

## Future Roadmap

#### 1️⃣ Speaker-Aware Extraction
- **Speaker Diarization**: Integrate `pyannote.audio` or `SpeechBrain` to segment clips per speaker
- **Speaker Queries**: Enable queries like "Give me all clips where speaker X is explaining something" or "Combine all funny reactions of speaker Y"
- **Speaker Embeddings**: Integrate speaker identity into LLM scoring for intelligent semantic merges based on who's speaking
- **Multi-speaker Analysis**: Track speaker transitions and dialogue patterns for better clip boundaries

#### 2️⃣ Emotion / Excitement Detection
- **Emotion Recognition**: Train or integrate pre-trained models for emotion/intensity detection beyond generic audio peaks
- **Engagement Scoring**: Guide LLM to rank clips not only by volume/motion but by perceived emotional engagement
- **Sentiment Analysis**: Combine audio emotion with transcript sentiment for deeper understanding
- **Facial Expression Analysis**: Detect smiles, laughter, surprise in video frames to enhance excitement scoring

#### 3️⃣ Adaptive Clip Duration
- **Platform Presets**: User-specified clip length preferences (short for TikTok/Reels, longer for podcasts/YouTube)
- **Intelligent Merging**: LLM can merge multiple peaks while respecting target duration constraints
- **Dynamic Segmentation**: Automatically adjust clip boundaries based on content density and pacing
- **Custom Templates**: Save and reuse clip duration strategies for different content types

#### 4️⃣ Content-Type Tuning
- **Auto-Classification**: Detect content type (comedy, sports, interview, tutorial, etc.) and adjust fusion weights accordingly
  - Stand-up comedy → Audio-heavy (0.7 audio, 0.3 video)
  - Sports/Gaming → Video-heavy (0.3 audio, 0.7 video)
  - Interviews/Podcasts → Balanced (0.5 audio, 0.5 video)
- **Genre-Specific Models**: Fine-tune excitement scoring for different video genres
- **Context-Aware Features**: Enable/disable specific features based on content type

#### 5️⃣ Real-Time / Streaming Mode
- **Live Stream Support**: Extract highlights on-the-fly from live streams or ongoing recordings
- **Streaming Inference**: Fast scoring models optimized for real-time processing
- **Incremental LLM Prompts**: Streaming-friendly LLM prompt design for progressive clip selection
- **Buffer Management**: Smart windowing for continuous audio/video analysis

#### 6️⃣ Auto-Subtitle / Captioning Integration
- **Forced Alignment**: Combine transcript with precise word-level timestamps
- **Subtitle Generation**: Auto-generate SRT/VTT files for each extracted clip
- **Multi-Language Support**: Transcribe and caption in multiple languages
- **Styling Options**: Customizable subtitle appearance for different platforms (TikTok, YouTube Shorts, Instagram)
- **Accessibility**: Ensure all clips are accessible with proper closed captions

### 🚀 Community Contributions Welcome!

We're excited about these features and welcome contributions! If you're interested in implementing any of these enhancements, please:
1. Open an issue to discuss your approach
2. Fork the repository and create a feature branch
3. Submit a pull request with comprehensive tests

## License

[MIT License](LICENSE) - see the LICENSE file for details.

## Contributing

Contributions welcome! Please read our [Contributing Guidelines](doc/CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## Acknowledgments

- YOLOv8 by Ultralytics
- CLIP by OpenAI
- Whisper by OpenAI
- OpenRouter for LLM API access
