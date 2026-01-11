# Quick Start Guide

Get up and running with Clipz in 5 minutes! This guide covers installation, first run, and basic usage.

> 📖 **For detailed documentation**, see the [full README](../README.md)

## What You'll Need

- Python 3.11 installed
- FFmpeg installed ([Download here](https://ffmpeg.org/download.html))
- OpenRouter API key ([Get free key](https://openrouter.ai/keys))

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Jit-Roy/Clipz.git
cd Clipz
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will download all required packages including PyTorch, Whisper, YOLO, and CLIP models.

### 4. Configure API Key

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

> 💡 **Tip:** Copy from `.env.example` if available

## Your First Clip Extraction

### Basic Command

```bash
python main.py path/to/your/video.mp4
```

**What happens:**
1. ✅ Extracts audio from video
2. ✅ Analyzes audio for excitement (loudness, rhythm, laughter, etc.)
3. ✅ Analyzes video for visual interest (motion, faces, composition)
4. ✅ Transcribes speech with timestamps
5. ✅ Uses AI (GPT-4o-mini) to intelligently select and merge clips
6. ✅ Exports clips to `output/clips_<timestamp>/`

**First run takes longer** (downloads models, no cache). Subsequent runs are much faster!

## Common Usage Examples

### Example 1: Extract Funny Moments
```bash
python main.py video.mp4 --query "give me 5 funny clips"
```

### Example 2: Podcast Highlights (Audio-Focused)
```bash
python main.py podcast.mp4 --audio-weight 0.7 --video-weight 0.3
```

### Example 3: Sports/Action (Visual-Focused)
```bash
python main.py sports.mp4 --audio-weight 0.3 --video-weight 0.7
```

### Example 4: Short Clips for TikTok
```bash
python main.py video.mp4 --min-duration 5 --max-duration 15 --query "give me 10 viral moments"
```

### Example 5: Faster Processing
```bash
python main.py video.mp4 --fps 1
```
Lower FPS = faster analysis (less accurate)

## Using the Python API

If you want to integrate Clipz into your own Python scripts:

```python
from main import ViralClipExtractor

# Initialize
extractor = ViralClipExtractor(
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

# Access results
for clip in results["clips"]:
    print(f"⏱️  {clip['start']:.1f}s - {clip['end']:.1f}s")
    print(f"📝 {clip['transcript'][:100]}...")
    print(f"⭐ Score: {clip['llm_interest_score']}/10")
    print(f"💡 {clip['reason']}")
    print()
```

> 📖 **For full API documentation**, see [README.md](../README.md#api-reference)

## Understanding the Output

After processing, your files are organized as follows:

### 📁 `output/clips_<timestamp>/`
**Your exported video clips** (ready to upload!)
```
clip_001.mp4
clip_002.mp4
clip_003.mp4
...
```

### 📁 `.cache/` (Hidden folder with metadata)

#### `.cache/metadata/`
JSON files for each clip with details:
```json
{
  "clip_number": 1,
  "start_time": 45.2,
  "end_time": 58.7,
  "duration": 13.5,
  "transcript": "...",
  "interest_score": 9.5,
  "reason": "Emotional storytelling with dramatic pause",
  "tags": ["emotional", "dramatic"]
}
```

#### `.cache/analysis/`
Complete analysis report for the entire video

#### `.cache/audio/` and `.cache/video/`
Cached features for faster re-processing

> 💡 **Tip:** The `.cache/` folder makes re-runs much faster! Delete it to force fresh analysis.

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Models downloading on first run | Normal! YOLO (~6MB) and CLIP models download automatically. |
| `FFmpeg not found` | Install FFmpeg and add to system PATH |
| `OpenRouter API error` | Verify `.env` file has valid `OPENROUTER_API_KEY` |
| `Out of memory` | Use `--fps 1` or process shorter videos |
| Slow processing | First run is slow (downloads models). Use caching for re-runs. |

## Next Steps

Now that you've run your first extraction, explore more:

- 📖 **[Full Documentation](../README.md)** - API reference, architecture, advanced features
- 🤝 **[Contributing Guide](CONTRIBUTING.md)** - Help improve Clipz
- 🔬 **Test Individual Modules**:
  - `python Audio/audio.py audio.wav` - Test audio analysis
  - `python video/video.py video.mp4` - Test video analysis
  - `python Transcription/transcribe.py audio.wav` - Test transcription

## Pro Tips

✅ **First run is slow** - Models download, no cache (~10-15 min for 10-min video)  
✅ **Re-runs are fast** - Cached features make it 3-5x faster  
✅ **GPU recommended** - Install CUDA PyTorch for 2-3x speedup  
✅ **Adjust weights** - Podcasts need high audio weight, sports need high video weight  
✅ **Natural language queries** - "Extract emotional moments", "Give me exciting gameplay", etc.  
✅ **Check `.cache/metadata/`** - See why each clip was selected

---

**Happy clipping !!!**  
Questions? Check the [README](../README.md) or open an issue on GitHub.
