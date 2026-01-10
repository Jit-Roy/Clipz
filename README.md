# Viral Clip Extractor

An AI-powered multimodal video clip extraction system that identifies viral-worthy moments using audio analysis, visual analysis, speech transcription, and LLM-based semantic understanding.

## Quick Start

👉 **New to this project?** Check out the [Quick Start Guide](QUICKSTART.md) for a 5-minute setup!

## Features

- 🎵 **Audio Analysis**: Multi-scale loudness, spectral novelty, rhythm detection, prosody analysis
- 🎬 **Video Analysis**: Motion tracking, semantic surprise (CLIP), composition scoring, shot detection
- 🗣️ **Transcription**: Automatic speech-to-text with timestamps using Whisper
- 🤖 **LLM Intelligence**: Smart clip selection and merging using GPT-4
- ⚡ **Parallel Processing**: Fast multi-threaded feature extraction
- 💾 **Caching**: Intelligent feature caching for faster re-runs

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg installed and available in PATH
- OpenRouter API key (for LLM features)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Clip_Extract

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (if not included)
# The yolov8n.pt model should be placed in the models/ folder
# It will be downloaded automatically on first use if not present
```

### Environment Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

## Quick Start

### Command Line Usage

```bash
# Basic usage - extract 10 clips
python main.py path/to/video.mp4

# Custom query
python main.py video.mp4 --query "give me 5 funny moments"

# Adjust weights
python main.py video.mp4 --audio-weight 0.7 --video-weight 0.3

# Custom settings
python main.py video.mp4 --fps 3 --min-duration 10 --max-duration 45
```

### Python API Usage

```python
from main import ClipExtractor

# Initialize extractor
extractor = ClipExtractor(
    audio_weight=0.5,
    video_weight=0.5,
    use_cache=True,
    output_dir="output"
)

# Process video
results = extractor.process(
    video_path="video.mp4",
    user_query="give me 10 interesting clips",
    target_fps=2,
    min_duration=5,
    max_duration=60,
    export=True
)

# Access results
for clip in results["clips"]:
    print(f"Clip: {clip['start']:.1f}s - {clip['end']:.1f}s")
    print(f"Transcript: {clip['transcript']}")
    print(f"Interest Score: {clip['llm_interest_score']}/10")
```

## Modules

### Audio Analysis (`audio.py`)

Analyzes audio for excitement signals:
- Loudness and energy analysis
- Spectral novelty detection (MFCC)
- Rhythm and onset detection
- Prosody and emotion analysis
- Silence contrast detection
- Structural boundary detection

```python
from audio import ClipAudio

detector = ClipAudio(sr=16000)
timestamps, scores = detector.compute_audio_scores("audio.wav")
```

### Video Analysis (`video.py`)

Analyzes video for visual excitement:
- Optical flow motion analysis
- Semantic surprise using CLIP embeddings
- Composition scoring (rule of thirds)
- Shot boundary detection
- Face detection and tracking
- Temporal rhythm analysis

```python
from video import ClipVideo

detector = ClipVideo()
timestamps, scores = detector.compute_visual_scores("video.mp4", target_fps=2)
```

### Transcription (`transcribe.py`)

Speech-to-text with timestamps:

```python
from transcribe import Transcriber

segments = Transcriber.transcribe_with_timestamps("audio.wav")
for seg in segments:
    print(f"[{seg['start']:.2f}s → {seg['end']:.2f}s] {seg['text']}")
```

### LLM Integration (`llm.py`)

Smart clip analysis and selection:

```python
from llm import LLM

llm = LLM()
response = llm.generate_text(
    prompt="Analyze these video clips...",
    model="openai/gpt-4o-mini"
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

### YOLO Model Not Found

Ensure `yolov8n.pt` is in the `models/` folder. The model will be downloaded automatically on first use if the ultralytics package is installed correctly.

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

## License

[MIT License](LICENSE) - see the LICENSE file for details.

## Contributing

Contributions welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## Acknowledgments

- YOLOv8 by Ultralytics
- CLIP by OpenAI
- Whisper by OpenAI
- OpenRouter for LLM API access
