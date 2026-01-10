"""
Viral Clip Extractor - Main Pipeline
Integrates audio, video, transcription, and LLM-based intelligent clip selection
"""

import os
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import subprocess
from tqdm import tqdm
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import custom modules
from audio import ClipAudio
from video import ClipVideo
from transcribe import Transcriber
from llm import LLM


class ClipExtractor:
    """
    Main pipeline for extracting viral clips using multi-modal analysis.
    """
    
    def __init__(
        self,
        audio_weight=0.5,
        video_weight=0.5,
        use_cache=True,
        output_dir="output"
    ):
        """
        Initialize the clip extractor.
        
        Args:
            audio_weight: Weight for audio scores (0-1)
            video_weight: Weight for video scores (0-1)
            use_cache: Whether to use cached features
            output_dir: Directory for output clips
        """
        self.audio_weight = audio_weight
        self.video_weight = video_weight
        self.use_cache = use_cache
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("[INIT] Initializing Viral Clip Extractor...")
        self.audio_detector = ClipAudio(sr=16000)  
        self.video_detector = ClipVideo()
        self.llm = LLM()
        print("[INIT] All models loaded successfully!\n")
    
    def _validate_video_path(self, video_path):
        """Validate and correct video path, checking videos/ folder if needed."""
        if not os.path.exists(video_path):
            # Try in videos/ subdirectory
            videos_path = os.path.join('videos', video_path)
            if os.path.exists(videos_path):
                print(f"[INFO] Found video in videos/ folder: {videos_path}")
                return videos_path
            else:
                raise FileNotFoundError(
                    f"Video file not found: {video_path}\n"
                    f"Also checked: {videos_path}\n"
                    f"Please provide full path or place video in videos/ folder"
                )
        return video_path
    
    def _extract_audio_from_video(self, video_path):
        """Extract audio track from video using FFmpeg."""
        # video_path is already validated by _validate_video_path
        audio_path = str(Path(video_path).with_suffix('.wav'))
        
        if os.path.exists(audio_path):
            print(f"[AUDIO] Audio file already exists: {audio_path}")
            return audio_path
        
        print(f"[AUDIO] Extracting audio from video...")
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # Match audio detector sample rate (optimized for speed)
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"[AUDIO] Audio extracted to: {audio_path}\n")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg failed to extract audio: {error_msg}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to your system PATH.")
        
        # Verify audio file was created
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio extraction failed: {audio_path} was not created")
        
        return audio_path
    
    def extract_features(self, video_path, target_fps=2):
        """
        Extract features from audio, video, and transcription in parallel.
        
        Args:
            video_path: Path to input video (already validated)
            target_fps: FPS for video analysis
            
        Returns:
            Dictionary with all extracted features
        """
        # video_path is already validated by caller
        
        # Extract audio from video
        audio_path = self._extract_audio_from_video(video_path)
        
        print("=" * 60)
        print("PHASE 1: MULTI-MODAL FEATURE EXTRACTION (PARALLEL)")
        print("=" * 60 + "\n")
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._extract_audio_features, audio_path
                ): "audio",
                executor.submit(
                    self._extract_video_features, video_path, target_fps
                ): "video",
                executor.submit(
                    self._extract_transcription, audio_path
                ): "transcription"
            }
            
            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=3, desc="Feature Extraction"):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                    print(f"✓ {task_name.capitalize()} features extracted")
                except Exception as e:
                    print(f"✗ {task_name.capitalize()} extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results[task_name] = None
        
        print(f"\n{'=' * 60}")
        print("PHASE 1 COMPLETE")
        print(f"{'=' * 60}\n")
        
        return results
    
    def _extract_audio_features(self, audio_path):
        """Extract audio excitement scores."""
        print("[AUDIO] Starting audio analysis...")
        timestamps, scores = self.audio_detector.compute_audio_scores(
            audio_path,
            use_cache=self.use_cache
        )
        return {"timestamps": timestamps, "scores": scores}
    
    def _extract_video_features(self, video_path, target_fps):
        """Extract video excitement scores."""
        print("[VIDEO] Starting video analysis...")
        timestamps, scores = self.video_detector.compute_visual_scores(
            video_path,
            target_fps=target_fps,
            use_cache=self.use_cache
        )
        return {"timestamps": timestamps, "scores": scores}
    
    def _extract_transcription(self, audio_path):
        """Extract transcription with timestamps."""
        print("[TRANSCRIPTION] Starting transcription...")
        segments = Transcriber.transcribe_with_timestamps(
            audio_path,
            model_size="base",
            verbose=False
        )
        return segments
    
    def fuse_signals(self, audio_data, video_data):
        """
        Fuse audio and video excitement scores into unified timeline.
        
        Args:
            audio_data: Dict with audio timestamps and scores
            video_data: Dict with video timestamps and scores
            
        Returns:
            Unified timestamps and combined scores
        """
        print("=" * 60)
        print("PHASE 2: SIGNAL FUSION")
        print("=" * 60 + "\n")
        
        # Validate input data
        if not audio_data or not isinstance(audio_data, dict):
            raise ValueError("Audio data is missing or invalid")
        if not video_data or not isinstance(video_data, dict):
            raise ValueError("Video data is missing or invalid")
        
        if "timestamps" not in audio_data or "scores" not in audio_data:
            raise ValueError("Audio data missing 'timestamps' or 'scores' key")
        if "timestamps" not in video_data or "scores" not in video_data:
            raise ValueError("Video data missing 'timestamps' or 'scores' key")
        
        if len(audio_data["timestamps"]) == 0:
            raise ValueError("Audio timestamps array is empty")
        if len(video_data["timestamps"]) == 0:
            raise ValueError("Video timestamps array is empty")
        
        # Determine common timeline (1 Hz = 1 score per second)
        max_time = max(audio_data["timestamps"][-1], video_data["timestamps"][-1])
        unified_timestamps = np.arange(0, max_time, 1.0)
        
        # Interpolate audio scores to unified timeline
        audio_interp = interp1d(
            audio_data["timestamps"],
            audio_data["scores"],
            kind='linear',
            bounds_error=False,
            fill_value=0
        )
        audio_unified = audio_interp(unified_timestamps)
        
        # Interpolate video scores to unified timeline
        video_interp = interp1d(
            video_data["timestamps"],
            video_data["scores"],
            kind='linear',
            bounds_error=False,
            fill_value=0
        )
        video_unified = video_interp(unified_timestamps)
        
        # Combine scores with weighted average
        combined_scores = (
            self.audio_weight * audio_unified +
            self.video_weight * video_unified
        )
        
        print(f"[FUSION] Timeline unified to {len(unified_timestamps)} points")
        print(f"[FUSION] Audio weight: {self.audio_weight}, Video weight: {self.video_weight}")
        print(f"\n{'=' * 60}")
        print("PHASE 2 COMPLETE")
        print(f"{'=' * 60}\n")
        
        return unified_timestamps, combined_scores, audio_unified, video_unified
    
    def generate_candidate_clips(
        self,
        timestamps,
        scores,
        transcript_segments,
        min_duration=5,
        max_duration=60,
        prominence=0.5,
        min_distance=10
    ):
        """
        Generate candidate clips using peak detection and transcript alignment.
        
        Args:
            timestamps: Unified timeline
            scores: Combined excitement scores
            transcript_segments: List of transcript segments
            min_duration: Minimum clip duration in seconds
            max_duration: Maximum clip duration in seconds
            prominence: Peak prominence threshold
            min_distance: Minimum distance between peaks (seconds)
            
        Returns:
            List of candidate clip dictionaries
        """
        print("=" * 60)
        print("PHASE 3: CANDIDATE CLIP GENERATION")
        print("=" * 60 + "\n")
        
        # Find peaks in combined scores
        peaks, properties = find_peaks(
            scores,
            prominence=prominence,
            distance=min_distance
        )
        
        print(f"[PEAKS] Detected {len(peaks)} excitement peaks")
        
        candidates = []
        
        for idx, peak_idx in enumerate(peaks):
            peak_time = timestamps[peak_idx]
            
            # Initial window around peak
            initial_start = max(0, peak_time - min_duration / 2)
            initial_end = min(timestamps[-1], peak_time + min_duration / 2)
            
            # Snap to transcript sentence boundaries
            aligned_clip = self._align_to_transcript_boundaries(
                initial_start,
                initial_end,
                transcript_segments,
                min_duration,
                max_duration
            )
            
            if aligned_clip:
                # Calculate average scores for this clip
                start_idx = np.searchsorted(timestamps, aligned_clip["start"])
                end_idx = np.searchsorted(timestamps, aligned_clip["end"])
                
                candidates.append({
                    "clip_id": idx + 1,
                    "start": aligned_clip["start"],
                    "end": aligned_clip["end"],
                    "duration": aligned_clip["end"] - aligned_clip["start"],
                    "transcript": aligned_clip["transcript"],
                    "peak_time": peak_time,
                    "avg_score": np.mean(scores[start_idx:end_idx]),
                    "peak_score": scores[peak_idx]
                })
        
        # Sort by peak score
        candidates.sort(key=lambda x: x["peak_score"], reverse=True)
        
        print(f"[CANDIDATES] Generated {len(candidates)} candidate clips")
        print(f"\n{'=' * 60}")
        print("PHASE 3 COMPLETE")
        print(f"{'=' * 60}\n")
        
        return candidates
    
    def _align_to_transcript_boundaries(
        self,
        start_time,
        end_time,
        transcript_segments,
        min_duration,
        max_duration
    ):
        """
        Align clip boundaries to complete sentences.
        Never cut mid-sentence.
        """
        # Find overlapping transcript segments
        overlapping = [
            seg for seg in transcript_segments
            if not (seg["end"] < start_time or seg["start"] > end_time)
        ]
        
        if not overlapping:
            return None
        
        # Snap to sentence boundaries
        aligned_start = overlapping[0]["start"]
        aligned_end = overlapping[-1]["end"]
        
        # Check duration constraints
        duration = aligned_end - aligned_start
        
        if duration < min_duration:
            # Try to extend
            # Find previous segment
            prev_idx = transcript_segments.index(overlapping[0]) - 1
            if prev_idx >= 0:
                aligned_start = transcript_segments[prev_idx]["start"]
        
        if duration > max_duration:
            # Trim to max duration while respecting sentence boundaries
            cumulative_duration = 0
            valid_segments = []
            
            for seg in overlapping:
                seg_duration = seg["end"] - seg["start"]
                if cumulative_duration + seg_duration <= max_duration:
                    valid_segments.append(seg)
                    cumulative_duration += seg_duration
                else:
                    break
            
            if valid_segments:
                aligned_start = valid_segments[0]["start"]
                aligned_end = valid_segments[-1]["end"]
            else:
                return None
        
        # Combine transcript text
        transcript_text = " ".join([seg["text"] for seg in overlapping])
        
        return {
            "start": aligned_start,
            "end": aligned_end,
            "transcript": transcript_text
        }
    
    def llm_analysis(
        self,
        candidates,
        user_query,
        audio_scores,
        video_scores,
        top_n_for_llm=30
    ):
        """
        Use LLM to analyze, merge, and rank clips based on user query.
        
        Args:
            candidates: List of candidate clips
            user_query: User's request (e.g., "give me 10 interesting clips")
            audio_scores: Audio excitement scores (for context)
            video_scores: Video excitement scores (for context)
            top_n_for_llm: Number of candidates to send to LLM
            
        Returns:
            Final selected clips with LLM scoring and merging
        """
        print("=" * 60)
        print("PHASE 4: LLM-BASED SEMANTIC ANALYSIS")
        print("=" * 60 + "\n")
        
        # Pre-filter to top N to avoid context window issues
        candidates_for_llm = candidates[:top_n_for_llm]
        
        print(f"[LLM] Analyzing top {len(candidates_for_llm)} candidates")
        print(f"[LLM] User query: '{user_query}'")
        
        # Prepare context for LLM
        llm_context = self._prepare_llm_context(candidates_for_llm, audio_scores, video_scores)
        
        # Create prompt
        prompt = self._create_llm_prompt(llm_context, user_query)
        
        # Get LLM response
        try:
            llm_response = self.llm.generate_text(
                prompt,
                model="openai/gpt-4o-mini",
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse LLM response
            final_clips = self._parse_llm_response(llm_response, candidates_for_llm)
            
            print(f"[LLM] Selected {len(final_clips)} final clips")
            
        except Exception as e:
            print(f"[LLM] Analysis failed: {e}")
            print("[LLM] Falling back to score-based ranking")
            final_clips = self._fallback_ranking(candidates_for_llm, user_query)
        
        print(f"\n{'=' * 60}")
        print("PHASE 4 COMPLETE")
        print(f"{'=' * 60}\n")
        
        return final_clips
    
    def _prepare_llm_context(self, candidates, audio_scores, video_scores):
        """Prepare structured context for LLM."""
        context = []
        
        for clip in candidates:
            context.append({
                "clip_id": clip["clip_id"],
                "start": round(clip["start"], 2),
                "end": round(clip["end"], 2),
                "duration": round(clip["duration"], 2),
                "transcript": clip["transcript"][:200],  # Truncate for context
                "excitement_score": round(clip["avg_score"], 3)
            })
        
        return context
    
    def _create_llm_prompt(self, context, user_query):
        """Create prompt for LLM clip selection."""
        prompt = f"""You are a viral video clip curator. Analyze these video segments and select the best clips based on the user's request.

        USER REQUEST: {user_query}

        CANDIDATE CLIPS:
        {json.dumps(context, indent=2)}

        INSTRUCTIONS:
        1. Analyze each clip's transcript for interestingness, emotion, humor, or viral potential
        2. Identify clips that should be MERGED (if they're adjacent and part of the same story)
        3. Score each clip or merged clip for interest (0-10)
        4. Return the requested number of clips, ranked by interest

        MERGING RULES:
        - Merge clips if they're within 5 seconds of each other
        - Merge if they continue the same topic/story
        - Merged clips should have combined start/end times

        OUTPUT FORMAT (JSON only, no other text):
        [
            {{
                "merged_clip_ids": [1, 2],
                "final_start": 10.5,
                "final_end": 45.0,
                "reason": "Why this clip is interesting",
                "interest_score": 9.5,
                "tags": ["emotional", "funny"]
            }}
        ]

        Return ONLY valid JSON, nothing else."""
        
        return prompt
    
    def _parse_llm_response(self, response, candidates):
        """Parse LLM JSON response and apply merges."""
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            json_str = response[json_start:json_end]
            
            llm_clips = json.loads(json_str)
            
            final_clips = []
            
            for llm_clip in llm_clips:
                # Find original candidate clips
                merged_ids = llm_clip.get("merged_clip_ids", [])
                
                if merged_ids:
                    # Use LLM's merged timestamps
                    clip_data = {
                        "start": llm_clip["final_start"],
                        "end": llm_clip["final_end"],
                        "duration": llm_clip["final_end"] - llm_clip["final_start"],
                        "transcript": " ".join([
                            c["transcript"] for c in candidates if c["clip_id"] in merged_ids
                        ]),
                        "llm_interest_score": llm_clip["interest_score"],
                        "reason": llm_clip["reason"],
                        "tags": llm_clip.get("tags", []),
                        "merged_from": merged_ids
                    }
                else:
                    clip_data = llm_clip
                
                final_clips.append(clip_data)
            
            return final_clips
            
        except Exception as e:
            print(f"[LLM] Failed to parse response: {e}")
            return []
    
    def _fallback_ranking(self, candidates, user_query):
        """Fallback to simple score-based ranking if LLM fails."""
        # Extract number from query if present
        import re
        match = re.search(r'\d+', user_query)
        n_clips = int(match.group()) if match else 5
        
        return candidates[:n_clips]
    
    def export_clips(self, video_path, clips, add_subtitles=False):
        """
        Export final clips as video files using FFmpeg.
        
        Args:
            video_path: Original video path
            clips: List of final clip dictionaries
            add_subtitles: Whether to burn subtitles into video
            
        Returns:
            List of exported clip file paths
        """
        if not clips:
            print("[EXPORT] No clips to export")
            return []
        
        print("=" * 60)
        print("PHASE 5: CLIP EXTRACTION & EXPORT")
        print("=" * 60 + "\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory for clips
        clips_dir = self.output_dir / f"clips_{timestamp}"
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache directory for metadata
        cache_dir = Path('.cache') / 'metadata'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        failed_clips = []
        
        print(f"[EXPORT] Extracting {len(clips)} clips from video...")
        print(f"[DEBUG] Video path: {video_path}")
        print(f"[DEBUG] Absolute video path: {os.path.abspath(video_path)}")
        
        for idx, clip in enumerate(tqdm(clips, desc="Exporting clips"), 1):
            output_file = clips_dir / f"clip_{idx:03d}.mp4"
            
            try:
                # Use absolute paths for FFmpeg
                video_abs_path = os.path.abspath(video_path)
                output_abs_path = os.path.abspath(output_file)
                
                # FFmpeg command for clip extraction
                cmd = [
                    'ffmpeg',
                    '-i', video_abs_path,
                    '-ss', str(clip['start']),
                    '-to', str(clip['end']),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-y',
                    output_abs_path
                ]
                
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                # Verify file was created
                if output_file.exists() and output_file.stat().st_size > 0:
                    exported_files.append(str(output_file))
                    
                    # Save metadata to cache folder
                    metadata_file = cache_dir / f"clip_{idx:03d}_{timestamp}.json"
                    clip_metadata = {
                        "clip_number": idx,
                        "video_file": str(output_file),
                        "start_time": clip['start'],
                        "end_time": clip['end'],
                        "duration": clip['duration'],
                        "transcript": clip.get('transcript', ''),
                        "interest_score": clip.get('llm_interest_score', 0),
                        "reason": clip.get('reason', ''),
                        "tags": clip.get('tags', [])
                    }
                    
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(clip_metadata, f, indent=2, ensure_ascii=False)
                else:
                    failed_clips.append(idx)
                    print(f"\n✗ Clip {idx} failed: File not created or empty")
                    
            except subprocess.CalledProcessError as e:
                failed_clips.append(idx)
                error_msg = e.stderr.decode() if e.stderr else str(e)
                # Show last 500 chars of error for better debugging
                print(f"\n✗ Clip {idx} failed: {error_msg[-500:]}")
            except Exception as e:
                failed_clips.append(idx)
                print(f"\n✗ Clip {idx} failed: {str(e)}")
        
        print(f"\n{'=' * 60}")
        print(f"[EXPORT] ✓ Successfully exported {len(exported_files)} clips")
        if failed_clips:
            print(f"[EXPORT] ✗ Failed clips: {failed_clips}")
        print(f"[EXPORT] Output folder: {clips_dir}")
        print(f"[EXPORT] Metadata cached in: {cache_dir}")
        print(f"{'=' * 60}\n")
        
        return exported_files
    
    def process(
        self,
        video_path,
        user_query="give me 10 interesting clips",
        target_fps=2,
        min_duration=5,
        max_duration=60,
        export=True
    ):
        """
        Complete end-to-end pipeline.
        
        Args:
            video_path: Path to input video
            user_query: User's clip request
            target_fps: FPS for video analysis
            min_duration: Minimum clip duration
            max_duration: Maximum clip duration
            export: Whether to export clips to files
            
        Returns:
            Dictionary with all results and metadata
        """
        print("\n" + "=" * 60)
        print("VIRAL CLIP EXTRACTOR - FULL PIPELINE")
        print("=" * 60 + "\n")
        print(f"Video: {video_path}")
        print(f"Query: {user_query}\n")
        
        # Validate and correct video path first
        video_path = self._validate_video_path(video_path)
        print(f"[INFO] Using video: {video_path}\n")
        
        # Phase 1: Feature Extraction
        features = self.extract_features(video_path, target_fps)
        
        # Validate features before proceeding
        if not features.get("audio"):
            raise RuntimeError("Audio feature extraction failed. Check audio file and audio.py module.")
        if not features.get("video"):
            raise RuntimeError("Video feature extraction failed. Check video file and video.py module.")
        if not features.get("transcription"):
            raise RuntimeError("Transcription failed. Check audio file and transcribe.py module.")
        
        # Phase 2: Signal Fusion
        unified_timestamps, combined_scores, audio_scores, video_scores = self.fuse_signals(
            features["audio"],
            features["video"]
        )
        
        # Phase 3: Candidate Generation
        candidates = self.generate_candidate_clips(
            unified_timestamps,
            combined_scores,
            features["transcription"],
            min_duration,
            max_duration
        )
        
        # Phase 4: LLM Analysis
        final_clips = self.llm_analysis(
            candidates,
            user_query,
            audio_scores,
            video_scores
        )
        
        # Phase 5: Export
        exported_files = []
        if export and final_clips:
            print(f"\n[DEBUG] Starting export of {len(final_clips)} clips...")
            exported_files = self.export_clips(video_path, final_clips)
            print(f"[DEBUG] Export returned {len(exported_files)} files")
        elif not final_clips:
            print("[WARNING] No clips generated. Check LLM response or lower thresholds.")
        
        # Compile results
        results = {
            "video_path": video_path,
            "user_query": user_query,
            "analysis_metadata": {
                "duration": float(unified_timestamps[-1]),
                "num_candidates": len(candidates),
                "num_final_clips": len(final_clips),
                "num_exported": len(exported_files),
                "audio_weight": self.audio_weight,
                "video_weight": self.video_weight
            },
            "clips": final_clips,
            "exported_files": exported_files
        }
        
        # Save overall analysis to cache folder
        cache_dir = Path('.cache') / 'analysis'
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = cache_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 60)
        if exported_files:
            clips_folder = self.output_dir / f"clips_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n✓ {len(exported_files)} video clips extracted successfully!")
            print(f"✓ Clips saved to: {clips_folder}")
        print(f"✓ Analysis metadata: {metadata_path}")
        print(f"✓ Total clips generated: {len(final_clips)}")
        print("=" * 60 + "\n")
        
        return results


# ================== EXAMPLE USAGE ==================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract viral clips from video using AI analysis")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--query", default="give me 10 interesting clips", help="Query for clip selection")
    parser.add_argument("--audio-weight", type=float, default=0.5, help="Weight for audio scores (0-1)")
    parser.add_argument("--video-weight", type=float, default=0.5, help="Weight for video scores (0-1)")
    parser.add_argument("--fps", type=int, default=2, help="Target FPS for video analysis")
    parser.add_argument("--min-duration", type=int, default=5, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=int, default=60, help="Maximum clip duration in seconds")
    parser.add_argument("--output-dir", default="output", help="Output directory for clips")
    parser.add_argument("--no-export", action="store_true", help="Skip exporting video files")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = ClipExtractor(
        audio_weight=args.audio_weight,
        video_weight=args.video_weight,
        use_cache=True,
        output_dir=args.output_dir
    )
    
    # Process video
    results = extractor.process(
        video_path=args.video_path,
        user_query=args.query,
        target_fps=args.fps,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        export=not args.no_export
    )
    
    # Print results
    print("\n📊 FINAL CLIPS:")
    for i, clip in enumerate(results["clips"], 1):
        print(f"\nClip {i}:")
        print(f"  Time: {clip['start']:.1f}s - {clip['end']:.1f}s ({clip['duration']:.1f}s)")
        print(f"  Transcript: {clip.get('transcript', 'N/A')[:100]}...")
        if 'llm_interest_score' in clip:
            print(f"  Interest Score: {clip['llm_interest_score']}/10")
        if 'reason' in clip:
            print(f"  Reason: {clip['reason']}")