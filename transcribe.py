import torch
import whisper
import json
import hashlib
from pathlib import Path

class Transcriber:
    @staticmethod
    def transcribe_with_timestamps(
        audio_path,
        model_size="base",
        verbose=False,
        use_cache=True
    ):
        """
        Transcribes audio and returns sentence-level timestamps.
        
        Args:
            audio_path: Path to audio file
            model_size: Whisper model size (tiny, base, small, medium, large)
            verbose: Print detailed output
            use_cache: Use cached transcription if available
        
        Returns:
            List of dicts:
            [
                {
                    "start": float,
                    "end": float,
                    "text": str
                },
                ...
            ]
        """
        
        # Check cache first
        if use_cache:
            cache_dir = Path('.cache') / 'transcription'
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate cache filename
            cache_key = f"{audio_path}_{model_size}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = cache_dir / f"transcript_{cache_hash}.json"
            
            if cache_file.exists():
                print(f"[TRANSCRIPTION] Using cached transcription")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        print(f"[TRANSCRIPTION] Transcribing audio (this may take a while)...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_size, device=device)

        result = model.transcribe(
            audio_path,
            task="transcribe",
            verbose=verbose
        )

        sentences = []

        for segment in result["segments"]:
            sentences.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
        
        # Save to cache
        if use_cache:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(sentences, f, indent=2, ensure_ascii=False)
            print(f"[TRANSCRIPTION] Cached transcription for future use")

        return sentences


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_path>")
        sys.exit(1)

    audio_file = sys.argv[1]

    output = Transcriber.transcribe_with_timestamps(audio_file)

    for s in output:
        print(f"[{s['start']:.2f}s → {s['end']:.2f}s] {s['text']}")
