import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow warnings before import
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import librosa
import scipy.signal as signal
from scipy.signal import find_peaks, convolve
from scipy.signal.windows import gaussian
import ruptures as rpt
import parselmouth
from parselmouth.praat import call
import hashlib
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import tensorflow_hub as hub
import soundfile as sf


class ClipAudio:
    """
    Professional audio-based viral clip detection system using multi-scale acoustic analysis.
    
    Features:
        - Multi-resolution loudness & energy analysis
        - Spectral novelty detection via MFCC/mel-spectrogram
        - Rhythm and onset detection
        - Prosody & emotion analysis (pitch, variance, rate)
        - Silence contrast and dramatic pauses
        - Expectation violation (second derivative)
        - Structural boundary detection
        - Semantic audio event classification (laughter, applause, screams)
        - Attention persistence modeling
        - Cross-scale temporal consistency
    
    Example:
        >>> detector = ClipAudio()
        >>> timestamps, scores, clips = detector.generate_clips(
        ...     audio_path="podcast.wav",
        ...     min_len=3,
        ...     threshold=0.15
        ... )
        >>> detector.plot_results(timestamps, scores, clips)
    """
    
    def __init__(self, sr=16000):
        """
        Initialize audio detector with pre-trained models.
        
        Args:
            sr: Sample rate for audio processing (default: 16000, optimized for speed)
        """
        self.sr = sr
        print("[INIT] Loading audio models...")
        
        # Load YAMNet for semantic audio events
        try:
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            print("[INIT] YAMNet loaded successfully")
        except Exception as e:
            print(f"[WARNING] YAMNet failed to load: {e}")
            self.yamnet_model = None
        
        print("[INIT] Audio detector ready!")
    
    # ================== UTILITIES ==================
    
    @staticmethod
    def robust_normalize(x):
        """Robust normalization using median absolute deviation."""
        x = np.array(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-6
        return np.clip((x - med) / mad, -3, 3)
    
    @staticmethod
    def audio_cache_name(audio_path):
        """Generate cache filename based on audio path."""
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join('.cache', 'audio')
        os.makedirs(cache_dir, exist_ok=True)
        
        h = hashlib.md5(audio_path.encode()).hexdigest()
        return os.path.join(cache_dir, f"audio_cache_{h}.npz")
    
    @staticmethod
    def ema_filter(signal_data, alpha=0.3):
        """Exponential moving average for temporal smoothing."""
        # Apply scipy.signal filtering for better performance
        b = [alpha]
        a = [1, -(1 - alpha)]
        ema = signal.lfilter(b, a, signal_data)
        return ema
    
    @staticmethod
    def multi_scale_window(signal_data, window_sizes_sec, hop_length, sr):
        """
        Compute multi-scale features across different time windows.
        
        Args:
            signal_data: Input signal array
            window_sizes_sec: List of window sizes in seconds
            hop_length: Hop length in samples
            sr: Sample rate
            
        Returns:
            Dictionary of features at each scale
        """
        features = {}
        for ws in window_sizes_sec:
            ws_frames = int(ws * sr / hop_length)
            if ws_frames > 1:
                kernel = np.ones(ws_frames) / ws_frames
                smoothed = convolve(signal_data, kernel, mode='same')
                features[f'{ws}s'] = smoothed
        return features
    
    @staticmethod
    def temporal_envelope(signal_data, window):
        """
        Compute slow-moving RMS envelope for perceived excitement.
        
        Uses human-attention window (~2 seconds) to suppress micro-spikes
        while keeping sustained excitement visible.
        
        Args:
            signal_data: Input signal array
            window: Window size in frames
            
        Returns:
            RMS envelope of the signal
        """
        squared = signal_data ** 2
        rms = np.sqrt(convolve(squared, np.ones(window)/window, mode='same'))
        return rms
    
    # ================== FEATURE EXTRACTION ==================
    
    def extract_loudness_energy(self, y, hop_length=512):
        """
        1. Loudness & Energy (multi-scale)
        
        Returns:
            Short-term and long-term RMS energy in dB
        """
        # Short-term (20-40ms)
        frame_length_short = int(0.03 * self.sr)
        rms_short = librosa.feature.rms(y=y, frame_length=frame_length_short, hop_length=hop_length)[0]
        
        # Long-term (250-500ms)
        frame_length_long = int(0.4 * self.sr)
        rms_long = librosa.feature.rms(y=y, frame_length=frame_length_long, hop_length=hop_length)[0]
        
        # Convert to dB
        energy_short = librosa.amplitude_to_db(rms_short + 1e-6, ref=np.max)
        energy_long = librosa.amplitude_to_db(rms_long + 1e-6, ref=np.max)
        
        return energy_short, energy_long
    
    def extract_spectral_novelty(self, y, hop_length=512):
        """
        2. Spectral Novelty - captures texture/speaker/music changes
        
        Returns:
            MFCC-based spectral novelty score
        """
        # Compute MFCCs with memory-efficient settings
        # Use lower n_fft to reduce memory usage for long audio
        n_fft = min(2048, len(y))
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
        
        # Compute frame-to-frame cosine distance
        novelty = np.zeros(mfcc.shape[1])
        window = 10  # Compare to past 10 frames
        
        for i in range(window, mfcc.shape[1]):
            curr = mfcc[:, i]
            prev_window = mfcc[:, i-window:i]
            
            # Mean of past window
            prev_mean = np.mean(prev_window, axis=1)
            
            # Cosine distance
            cos_sim = np.dot(curr, prev_mean) / (np.linalg.norm(curr) * np.linalg.norm(prev_mean) + 1e-6)
            novelty[i] = 1 - cos_sim
        
        return novelty
    
    def extract_rhythm_onsets(self, y, hop_length=512):
        """
        3. Temporal Rhythm & Onsets - beats, bursts, fast speech
        
        Returns:
            Onset strength and tempo-normalized rhythm
        """
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=hop_length)
        
        # Tempo
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr, hop_length=hop_length)[0]
        
        # Rhythm variance (short-term energy variance)
        rhythm_var = np.array([np.std(onset_env[max(0, i-10):i+1]) for i in range(len(onset_env))])
        
        return onset_env, rhythm_var, tempo
    
    def extract_prosody_emotion(self, audio_path):
        """
        4. Prosody & Emotion - pitch, variance, speaking rate
        
        Uses Praat/Parselmouth for robust pitch extraction
        
        Returns:
            Pitch contour, pitch variance, energy-pitch coupling
        """
        try:
            snd = parselmouth.Sound(audio_path)
            
            # Extract pitch
            pitch = snd.to_pitch(time_step=0.01)
            pitch_values = pitch.selected_array['frequency']
            pitch_values[pitch_values == 0] = np.nan  # Remove unvoiced
            
            # Pitch variance (excitement indicator)
            pitch_var = np.array([
                np.nanstd(pitch_values[max(0, i-10):i+1]) 
                for i in range(len(pitch_values))
            ])
            
            # Replace NaN with median
            pitch_values = np.nan_to_num(pitch_values, nan=np.nanmedian(pitch_values))
            pitch_var = np.nan_to_num(pitch_var, nan=0)
            
            return pitch_values, pitch_var
            
        except Exception as e:
            print(f"[WARNING] Prosody extraction failed: {e}")
            # Return zeros as fallback
            n_frames = int(len(librosa.load(audio_path, sr=self.sr)[0]) / 512)
            return np.zeros(n_frames), np.zeros(n_frames)
    
    def extract_silence_contrast(self, y, hop_length=512, top_db=40):
        """
        5. Silence & Contrast - dramatic pauses, punchline setup
        
        Returns:
            Silence mask and silence-to-sound transition scores
        """
        # Detect non-silent intervals
        intervals = librosa.effects.split(y, top_db=top_db, hop_length=hop_length)
        
        # Create silence mask
        n_frames = int(len(y) / hop_length) + 1
        silence_mask = np.ones(n_frames)
        
        for start, end in intervals:
            start_frame = int(start / hop_length)
            end_frame = int(end / hop_length)
            silence_mask[start_frame:end_frame] = 0
        
        # Detect transitions (silence → sound)
        transitions = np.diff(silence_mask, prepend=silence_mask[0])
        transition_score = np.abs(transitions)
        
        return silence_mask, transition_score
    
    def extract_structural_boundaries(self, y, hop_length=512):
        """
        7. Structural Boundaries - section changes, speaker switches
        
        Uses ruptures for change-point detection
        
        Returns:
            Boundary score array
        """
        try:
            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, hop_length=hop_length)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Change-point detection (optimized parameters for speed)
            algo = rpt.Pelt(model="rbf", min_size=20, jump=10).fit(mel_db.T)
            change_points = algo.predict(pen=15)
            
            # Create boundary score
            boundary_score = np.zeros(mel_db.shape[1])
            for cp in change_points[:-1]:  # Last point is always end
                if cp < len(boundary_score):
                    boundary_score[cp] = 1.0
            
            # Smooth boundaries
            kernel = gaussian(11, 2)
            boundary_score = convolve(boundary_score, kernel / kernel.sum(), mode='same')
            
            return boundary_score
            
        except Exception as e:
            print(f"[WARNING] Boundary detection failed: {e}")
            n_frames = int(len(y) / hop_length) + 1
            return np.zeros(n_frames)
    
    def extract_semantic_events(self, audio_path):
        """
        8. Semantic Audio Events - laughter, applause, screams
        
        Uses YAMNet for audio event classification
        
        Returns:
            Semantic excitement score based on event classes
        """
        if self.yamnet_model is None:
            print("[WARNING] YAMNet not available, skipping semantic events")
            y, _ = librosa.load(audio_path, sr=self.sr)
            n_frames = int(len(y) / 512) + 1
            return np.zeros(n_frames)
        
        try:
            # Load audio for YAMNet (16kHz requirement)
            y_16k, _ = librosa.load(audio_path, sr=16000)
            
            # Run YAMNet
            scores, embeddings, spectrogram = self.yamnet_model(y_16k)
            scores = scores.numpy()
            
            # Define high-excitement event classes (indices in YAMNet)
            # Laughter: 321, Applause: 138, Cheering: 139, Screaming: 422
            # Music: 137
            # Speech is baseline, not excitement - REMOVED
            excitement_classes = {
                321: 2.0,   # Laughter
                138: 1.8,   # Applause
                139: 1.8,   # Cheering
                422: 1.5,   # Screaming
                137: 1.0    # Music (optional, lower weight)
            }
            
            # Compute excitement score per frame
            excitement = np.zeros(scores.shape[0])
            for class_idx, weight in excitement_classes.items():
                excitement += scores[:, class_idx] * weight
            
            # Interpolate to match our hop_length
            y, _ = librosa.load(audio_path, sr=self.sr)
            n_frames = int(len(y) / 4096) + 1  # Match current hop_length
            excitement_interp = np.interp(
                np.linspace(0, len(excitement), n_frames),
                np.arange(len(excitement)),
                excitement
            )
            
            return excitement_interp
            
        except Exception as e:
            print(f"[WARNING] Semantic event extraction failed: {e}")
            y, _ = librosa.load(audio_path, sr=self.sr)
            n_frames = int(len(y) / 4096) + 1  # Match current hop_length
            return np.zeros(n_frames)
    
    def compute_cross_scale_consistency(self, signal_data, hop_length=512):
        """
        10. Cross-Scale Consistency - multi-resolution agreement
        
        Returns:
            Consistency score (higher = multiple scales agree)
        """
        # Define scales: 50ms, 250ms, 2s
        window_sizes = [0.05, 0.25, 2.0]
        scales = self.multi_scale_window(signal_data, window_sizes, hop_length, self.sr)
        
        # Normalize each scale
        normalized_scales = []
        for scale_key in scales:
            norm = self.robust_normalize(scales[scale_key])
            normalized_scales.append(norm)
        
        # Consistency = low variance across scales (they agree)
        consistency = 1.0 / (np.std(normalized_scales, axis=0) + 1.0)
        
        return consistency
    
    def audio_excitement_score(self, L_short, L_long, SN, Onset, Rhythm, Pitch_var, 
                               Silence_trans, EV, Boundary, Semantic, AP, CSC):
        """
        Final audio excitement scoring function.
        
        Weights optimized for human excitement detection:
        - Spectral novelty (25%): What changed matters most
        - Expectation violation (17%): Surprise drives engagement
        - Semantic events (12%): Laughter/applause = viral
        - Energy + rhythm + prosody (30% combined): Base excitement
        - Structure + silence + persistence (16%): Context modulation
        """
        return (
            0.10 * L_short +        # Short-term loudness
            0.08 * L_long +         # Long-term loudness
            0.25 * SN +             # Spectral Novelty (CRITICAL)
            0.08 * Onset +          # Onset strength
            0.07 * Rhythm +         # Rhythm variance
            0.05 * Pitch_var +      # Pitch variance (emotion)
            0.06 * Silence_trans +  # Silence contrast
            0.17 * EV +             # Expectation Violation (CRITICAL)
            0.04 * Boundary +       # Structural boundaries
            0.12 * Semantic +       # Semantic events (IMPORTANT)
            0.05 * AP +             # Attention persistence
            0.03 * CSC              # Cross-scale consistency
        )
    
    # ================== MAIN PIPELINE ==================
    
    def compute_audio_scores(self, audio_path, use_cache=False):
        """
        Compute audio excitement scores for entire audio file.
        
        Args:
            audio_path: Path to audio file
            use_cache: Whether to use cached results if available
            
        Returns:
            timestamps: Array of timestamps (seconds)
            scores: Array of audio excitement scores
        """
        cache_file = self.audio_cache_name(audio_path)
        
        if use_cache and os.path.exists(cache_file):
            print(f"[CACHE] Loading precomputed data: {cache_file}")
            data = np.load(cache_file)
            return data["timestamps"], data["scores"]
        
        # Show file size and loading progress
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"[PROCESSING] Loading & resampling audio ({file_size_mb:.1f} MB) to {self.sr}Hz...")
        
        # Load directly at target sample rate (faster than load + resample)
        import time
        start_time = time.time()
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        elapsed = time.time() - start_time
        print(f"[PROCESSING] Audio loaded in {elapsed:.1f}s")
        
        # Use larger hop_length for long audio to reduce memory usage
        # For a 10-minute audio at 22kHz, this gives ~600 frames instead of 12000+
        hop_length = 4096  # ~186ms hop = ~5.4 frames per second
        n_frames = int(len(y) / hop_length) + 1
        print(f"[INFO] Audio duration: {len(y)/self.sr:.1f}s, Processing {n_frames} frames")
        timestamps = librosa.frames_to_time(np.arange(n_frames), sr=self.sr, hop_length=hop_length)
        
        print("[PROCESSING] Extracting features...")
        
        # Feature extraction steps with progress tracking
        feature_steps = [
            ("Loudness & Energy", lambda: self.extract_loudness_energy(y, hop_length)),
            ("Spectral Novelty", lambda: (self.extract_spectral_novelty(y, hop_length),)),
            ("Rhythm & Onsets", lambda: self.extract_rhythm_onsets(y, hop_length)),
            # Skip prosody - parselmouth is very slow (~5 mins)
            # ("Prosody & Emotion", lambda: self.extract_prosody_emotion(audio_path)),
            ("Silence & Contrast", lambda: self.extract_silence_contrast(y, hop_length)),
            ("Structural Boundaries", lambda: (self.extract_structural_boundaries(y, hop_length),)),
            ("Semantic Events", lambda: (self.extract_semantic_events(audio_path),)),
        ]
        
        results = {}
        for step_name, step_func in tqdm(feature_steps, desc="Extracting Audio Features", unit="feature",
                                          mininterval=0.1, dynamic_ncols=True, leave=True):
            results[step_name] = step_func()
        
        # 1. Loudness & Energy
        L_short, L_long = results["Loudness & Energy"]
        
        # 2. Spectral Novelty
        SN = results["Spectral Novelty"][0]
        
        # 3. Rhythm & Onsets
        Onset, Rhythm, tempo = results["Rhythm & Onsets"]
        
        # 4. Prosody & Emotion - SKIPPED for speed
        Pitch_var = np.zeros(len(Onset))  # Dummy values
        
        # 5. Silence & Contrast
        Silence_mask, Silence_trans = results["Silence & Contrast"]
        
        # 7. Structural Boundaries
        Boundary = results["Structural Boundaries"][0]
        
        # 8. Semantic Events
        Semantic = results["Semantic Events"][0]
        
        # Align all features to same length
        min_len = min(len(L_short), len(SN), len(Onset), len(Rhythm), 
                     len(Pitch_var), len(Silence_trans), len(Boundary), len(Semantic))
        
        L_short = L_short[:min_len]
        L_long = L_long[:min_len]
        SN = SN[:min_len]
        Onset = Onset[:min_len]
        Rhythm = Rhythm[:min_len]
        Pitch_var = Pitch_var[:min_len]
        Silence_trans = Silence_trans[:min_len]
        Boundary = Boundary[:min_len]
        Semantic = Semantic[:min_len]
        Silence_mask = Silence_mask[:min_len]
        timestamps = timestamps[:min_len]
        
        # Normalize features
        print("[PROCESSING] Normalizing features...")
        L_short = self.robust_normalize(L_short)
        L_long = self.robust_normalize(L_long)
        SN = self.robust_normalize(SN)
        Onset = self.robust_normalize(Onset)
        Rhythm = self.robust_normalize(Rhythm)
        Pitch_var = self.robust_normalize(Pitch_var)
        Silence_trans = self.robust_normalize(Silence_trans)
        Boundary = self.robust_normalize(Boundary)
        Semantic = self.robust_normalize(Semantic)
        
        # Gate Silence_trans to only be meaningful during speech regions
        speech_mask = 1 - Silence_mask
        Silence_trans = Silence_trans * speech_mask
        
        # Compute raw scores (without EV and AP)
        raw_scores = (
            0.10 * L_short + 0.08 * L_long + 0.25 * SN + 0.08 * Onset +
            0.07 * Rhythm + 0.05 * Pitch_var + 0.06 * Silence_trans +
            0.04 * Boundary + 0.12 * Semantic
        )
        
        # Gate raw scores by speech regions (no speech → no excitement)
        raw_scores = raw_scores * speech_mask
        
        # 6. Expectation Violation (FIX 3: change relative to recent mean)
        print("[PROCESSING] Computing expectation violation...")
        local_mean = convolve(raw_scores, np.ones(50)/50, mode='same')
        EV = np.abs(raw_scores - local_mean)
        EV = self.robust_normalize(EV)
        
        # 9. Attention Persistence (FIX 4: only reward sustained peaks)
        print("[PROCESSING] Computing attention persistence...")
        threshold_ap = np.percentile(raw_scores, 75)
        AP = np.where(raw_scores > threshold_ap,
                      self.ema_filter(raw_scores, 0.3),
                      0)
        AP = self.robust_normalize(AP)
        
        # 10. Cross-Scale Consistency - SKIPPED for speed (expensive multi-scale convolution)
        # print("[PROCESSING] Computing cross-scale consistency...")
        # CSC = self.compute_cross_scale_consistency(raw_scores, hop_length)
        # CSC = self.robust_normalize(CSC)
        CSC = np.zeros(len(raw_scores))  # Dummy values
        
        # Final composite score
        scores = np.array([
            self.audio_excitement_score(
                L_short[i], L_long[i], SN[i], Onset[i], Rhythm[i], Pitch_var[i],
                Silence_trans[i], EV[i], Boundary[i], Semantic[i], AP[i], CSC[i]
            )
            for i in range(len(L_short))
        ])
        
        # FIX 5: Enforce baseline subtraction (excitement must rise above baseline)
        scores = scores - np.percentile(scores, 30)
        scores = np.clip(scores, 0, None)
        
        # Apply temporal envelope smoothing (human-attention window)
        print("[PROCESSING] Applying temporal envelope smoothing...")
        attention_window_sec = 2.0  # 2 seconds for human perception
        hop_duration = hop_length / self.sr
        window_frames = int(attention_window_sec / hop_duration)
        scores = self.temporal_envelope(scores, window_frames)
        
        # Save to cache
        np.savez(cache_file, timestamps=timestamps, scores=scores)
        print(f"[CACHE] Saved to {cache_file}")
        
        return timestamps, scores
    
    # ================== CLIP EXTRACTION ==================
    
    @staticmethod
    def extract_audio_clips(timestamps, scores, min_len=2, threshold=0.15, padding=1.5):
        """
        Extract viral clip segments from scores.
        
        Args:
            timestamps: Array of timestamps
            scores: Array of excitement scores
            min_len: Minimum clip length in seconds
            threshold: Peak prominence threshold
            padding: Padding around peaks in seconds
            
        Returns:
            List of (start, end) tuples in seconds
        """
        # Smooth scores
        window = gaussian(7, 1.5)
        smooth_scores = convolve(scores, window/window.sum(), mode='same')
        
        # Peak detection with prominence
        peaks, _ = find_peaks(smooth_scores, prominence=threshold)
        clips = []
        for idx in peaks:
            start = max(timestamps[idx] - padding, timestamps[0])
            end = min(timestamps[idx] + padding, timestamps[-1])
            clips.append((start, end))
        
        # Merge overlapping
        merged = []
        for s, e in sorted(clips):
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        
        return [(s, e) for s, e in merged if e - s >= min_len]
    
    def generate_clips(self, audio_path, min_len=2, threshold=0.15, use_cache=False):
        """
        Complete pipeline: analyze audio and extract viral clips.
        
        Args:
            audio_path: Path to audio file
            min_len: Minimum clip length in seconds
            threshold: Peak prominence threshold
            use_cache: Whether to use cached results
            
        Returns:
            timestamps: Array of timestamps
            scores: Array of excitement scores
            clips: List of (start, end) tuples for viral segments
        """
        timestamps, scores = self.compute_audio_scores(audio_path, use_cache=use_cache)
        clips = self.extract_audio_clips(timestamps, scores, min_len=min_len, threshold=threshold)
        return timestamps, scores, clips
    
    # ================== VISUALIZATION ==================
    
    @staticmethod
    def plot_results(timestamps, scores, clips, title="Audio Excitement Timeline"):
        """
        Plot excitement scores and highlight viral clips.
        
        Args:
            timestamps: Array of timestamps
            scores: Array of excitement scores
            clips: List of (start, end) tuples
            title: Plot title
        """
        plt.figure(figsize=(16, 5))
        plt.plot(timestamps, scores, label="Audio Excitement Score", linewidth=1.5, color='#2E86AB')
        for s, e in clips:
            plt.axvspan(s, e, color="red", alpha=0.3, label='Viral Clip' if s == clips[0][0] else '')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Excitement Score", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_clips(clips):
        """Print detected clips in formatted output."""
        print("\n🎵 VIRAL AUDIO CLIPS FOUND:")
        if not clips:
            print("No clips detected. Try lowering the threshold.")
            return
        for i, (s, e) in enumerate(clips, 1):
            print(f"Clip {i}: {s:.2f}s → {e:.2f}s (Duration: {e-s:.2f}s)")


# ================== EXAMPLE USAGE ==================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio.py <audio_path>")
        sys.exit(1)

    detector = ClipAudio(sr=16000)  
    audio_path = sys.argv[1]
    timestamps, scores, clips = detector.generate_clips(
        audio_path=audio_path,
        min_len=3,
        threshold=0.15,
        use_cache=False
    )
    detector.print_clips(clips)
    detector.plot_results(timestamps, scores, clips, title="Audio Excitement Timeline (Red = Viral Clips)")