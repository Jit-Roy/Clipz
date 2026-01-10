# Quick Start Guide

Get started with Viral Clip Extractor in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- FFmpeg installed ([Download here](https://ffmpeg.org/download.html))
- OpenRouter API key ([Get free key](https://openrouter.ai/keys))

## Installation

```bash
# 1. Clone or download the project
cd Clip_Extract

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Copy from `.env.example` if needed.

## First Run

```bash
# Test with a sample video
python main.py path/to/your/video.mp4
```

This will:
1. Extract audio from video
2. Analyze audio for excitement signals
3. Analyze video for visual excitement
4. Transcribe speech with timestamps
5. Use AI to select the best clips
6. Export clips to `output/clips_<timestamp>/`

## Basic Usage Examples

### Extract 5 clips
```bash
python main.py video.mp4 --query "give me 5 funny clips"
```

### Prioritize audio over video
```bash
python main.py video.mp4 --audio-weight 0.7 --video-weight 0.3
```

### Adjust clip duration
```bash
python main.py video.mp4 --min-duration 10 --max-duration 30
```

### Just analyze, don't export
```bash
python main.py video.mp4 --no-export
```

## Python API Usage

```python
from main import ClipExtractor

# Initialize
extractor = ClipExtractor(
    audio_weight=0.5,
    video_weight=0.5,
    output_dir="output"
)

# Process video
results = extractor.process(
    video_path="video.mp4",
    user_query="give me 10 interesting clips",
    export=True
)

# Print results
for clip in results["clips"]:
    print(f"Clip: {clip['start']:.1f}s - {clip['end']:.1f}s")
    print(f"Score: {clip['llm_interest_score']}/10")
    print(f"Reason: {clip['reason']}")
```

## Output

After processing, you'll find:

1. **Clips**: `output/clips_<timestamp>/clip_001.mp4`, etc.
2. **Metadata**: `output/clips_<timestamp>/clip_001.json` for each clip
3. **Analysis**: `output/analysis_<timestamp>.json` with complete results

## Troubleshooting

### "YOLO model not found"
The `yolov8n.pt` file should be in the `models/` folder. It's included in the project.

### "FFmpeg not found"
Install FFmpeg and ensure it's in your system PATH.

### "OpenRouter API error"
Check your `.env` file has a valid `OPENROUTER_API_KEY`.

### Out of memory
Try lowering FPS: `--fps 1` or process shorter videos.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Explore individual modules: `audio.py`, `video.py`, `transcribe.py`
- Test with your own videos in the `videos/` folder

## Tips

- First run is slower (downloads models, no cache)
- Subsequent runs are faster (uses cached features)
- GPU recommended but not required
- Works best with videos 1-30 minutes long
- Adjust weights based on your content type:
  - Podcasts: Higher audio weight
  - Action videos: Higher video weight
  - Interviews: Balanced weights

Happy clipping! 🎬
