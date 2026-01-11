import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import os
import hashlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve
from scipy.signal.windows import gaussian
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import dlib
import warnings
warnings.filterwarnings("ignore")


class ClipVideo:
    """
    Professional viral clip detection system using multi-modal visual analysis.
    
    Features:
        - Motion analysis via optical flow
        - Semantic surprise detection using CLIP embeddings
        - Composition scoring with rule-of-thirds
        - Rhythm and temporal dynamics
        - Expectation violation detection
        - Attention persistence tracking
        - Shot boundary detection
    
    Example:
        >>> detector = ClipVideo()
        >>> timestamps, scores, clips = detector.generate_clips(
        ...     video_path="video.mp4",
        ...     target_fps=2,
        ...     min_len=3,
        ...     threshold=0.1
        ... )
        >>> detector.plot_results(timestamps, scores, clips)
    """
    
    def __init__(self, model_dir="models"):
        """Initialize models and detectors.
        
        Args:
            model_dir: Directory containing YOLO model files (default: 'models')
        """
        print("[INIT] Loading models...")
        
        # Get absolute path to models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, model_dir, "yolov8n.pt")
        
        # YOLO will auto-download if model doesn't exist
        # Just pass the model name and ultralytics handles it
        if os.path.exists(model_path):
            print(f"[INIT] Loading YOLO from: {model_path}")
            self.yolo = YOLO(model_path)
        else:
            print(f"[INIT] YOLO model not found at {model_path}")
            print("[INIT] Downloading yolov8n.pt from Ultralytics (first run only)...")
            # Ultralytics will auto-download to its cache if we just pass the model name
            self.yolo = YOLO("yolov8n.pt")
            # Save to local models folder for future use
            os.makedirs(os.path.join(base_dir, model_dir), exist_ok=True)
            print(f"[INIT] YOLO model downloaded successfully!")
        
        # Use relative cache directory for Hugging Face models
        hf_cache_dir = os.path.join(base_dir, ".cache", "huggingface")
        os.makedirs(hf_cache_dir, exist_ok=True)
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=hf_cache_dir)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=True,                   
            cache_dir=hf_cache_dir
        )
        self.face_detector = dlib.get_frontal_face_detector()
        print("[INIT] Models loaded successfully!")
    
    # ================== UTILITIES ==================
    
    @staticmethod
    def robust_normalize(x):
        """Robust normalization using median absolute deviation."""
        x = np.array(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-6
        return np.clip((x - med) / mad, -3, 3)
    
    @staticmethod
    def rhythm_score(signal, window=10):
        """Calculate rhythm score from motion signal."""
        diff = np.abs(np.diff(signal))
        return np.std(diff[-window:]) if len(diff) >= window else 0
    
    @staticmethod
    def video_cache_name(video_path, target_fps):
        """Generate cache filename based on video path and fps."""
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join('.cache', 'video')
        os.makedirs(cache_dir, exist_ok=True)
        
        h = hashlib.md5((video_path + str(target_fps)).encode()).hexdigest()
        return os.path.join(cache_dir, f"visual_cache_{h}.npz")
    
    @staticmethod
    def ema_filter(signal, alpha=0.3):
        """Exponential moving average for temporal smoothing."""
        ema = [signal[0]]
        for val in signal[1:]:
            ema.append(alpha * val + (1 - alpha) * ema[-1])
        return np.array(ema)
    
    @staticmethod
    def iou(box1, box2):
        """Intersection over Union for bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    # ================== VISUAL FEATURES ==================
    
    @staticmethod
    def composition_score(detections, frame_shape):
        """Weighted composition: rule-of-thirds + object size."""
        h, w = frame_shape[:2]
        center = np.array([w / 2, h / 2])
        scores = []
        for x1, y1, x2, y2 in detections:
            obj_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            d_center = np.linalg.norm(obj_center - center) / np.linalg.norm(center)
            
            thirds_x = np.array([w / 3, 2 * w / 3])
            thirds_y = np.array([h / 3, 2 * h / 3])
            d_thirds = min([np.linalg.norm(obj_center - np.array([tx, ty])) 
                            for tx in thirds_x for ty in thirds_y]) / np.linalg.norm(center)
            
            size = ((x2 - x1) * (y2 - y1)) / (w * h)
            score = (1 - d_center) * 0.5 + (1 - d_thirds) * 0.3 + size * 0.2
            scores.append(score)
        return max(scores) if scores else 0
    
    @staticmethod
    def visual_surprise(curr_feat, prev_feats):
        """Semantic novelty using CLIP embeddings."""
        if curr_feat is None or not prev_feats:
            return 0.0
        sims = []
        curr_norm = np.linalg.norm(curr_feat)
        for f in prev_feats:
            fn = np.linalg.norm(f)
            if fn > 1e-6:
                sims.append(np.dot(curr_feat, f) / (curr_norm * fn))
        return 1 - np.mean(sims) if sims else 0.0
    
    def thumbnailability(self, frame, detections):
        """Sharpness + contrast + face/saliency."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = frame.std()
        
        faces = self.face_detector(gray, 1)
        face_factor = 1 + len(faces) * 0.5
        
        dominance = len(detections) == 1
        return sharpness * contrast * face_factor * (1.5 if dominance else 1.0)
    
    @staticmethod
    def track_dominant_object(curr_boxes, prev_dominant, frame_shape):
        """Track persistence of dominant object across frames."""
        if len(curr_boxes) == 0:
            return None, 0.0
        
        # Find largest object
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in curr_boxes]
        dominant_idx = np.argmax(areas)
        dominant_box = curr_boxes[dominant_idx]
        
        # Calculate persistence
        if prev_dominant is None:
            return dominant_box, 0.0
        
        persistence = ClipVideo.iou(dominant_box, prev_dominant)
        return dominant_box, persistence
    
    @staticmethod
    def visual_score(M, S, C, R, I, T, EV, AP, SB):
        """Enhanced scoring with new dimensions."""
        return (
            0.13 * M +      # Motion (reduced further)
            0.25 * S +      # Semantic surprise (INCREASED - most important)
            0.10 * C +      # Composition (reduced)
            0.06 * R +      # Rhythm (reduced)
            0.06 * I +      # Interest count (reduced)
            0.08 * T +      # Thumbnailability (reduced significantly)
            0.17 * EV +     # Expectation Violation (INCREASED)
            0.07 * AP +     # Attention Persistence (slight increase)
            0.08 * SB       # Shot Boundary (INCREASED)
        )
    
    # ================== MAIN PIPELINE ==================
    
    def compute_visual_scores(self, video_path, target_fps=1, use_cache=False):
        """
        Compute visual excitement scores for entire video.
        
        Args:
            video_path: Path to video file
            target_fps: Sampling rate for analysis (default: 1 FPS)
            use_cache: Whether to use cached results if available
            
        Returns:
            timestamps: Array of timestamps (seconds)
            scores: Array of visual excitement scores
        """
        cache_file = self.video_cache_name(video_path, target_fps)

        if use_cache and os.path.exists(cache_file):
            print(f"[CACHE] Loading precomputed data: {cache_file}")
            data = np.load(cache_file)
            return data["timestamps"], data["scores"]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(int(round(fps / target_fps)), 1)

        ret, prev = cap.read()
        if not ret:
            raise RuntimeError("Could not read video")

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        timestamps = []
        M, S, C, R, I, T = [], [], [], [], [], []
        SB = []  # Shot boundary
        AP = []  # Attention persistence
        
        prev_feats = []
        flow_signal = []
        dominant_history = []
        prev_dominant = None
        
        t = 0.0
        frame_idx = 1
        pbar = tqdm(total=total_frames, desc=f"Processing ({target_fps} FPS)", unit="frame", 
                    mininterval=0.1, dynamic_ncols=True, leave=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)

            if frame_idx % step != 0:
                frame_idx += 1
                t += 1 / fps
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---- Shot Boundary Detection ----
            frame_diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            SB.append(frame_diff)

            # ---- Motion ----
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=2).mean()
            M.append(mag)
            flow_signal.append(mag)

            # ---- Semantic Surprise (CLIP) ----
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.clip_processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                feat = self.clip_model.get_image_features(**inputs).cpu().numpy().flatten()
            S.append(self.visual_surprise(feat, prev_feats[-10:]))
            prev_feats.append(feat)

            # ---- YOLO detections ----
            results = self.yolo(frame, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            
            C.append(self.composition_score(boxes, frame.shape))
            I.append(len(boxes))
            T.append(self.thumbnailability(frame, boxes))

            # ---- Attention Persistence ----
            curr_dominant, persistence = self.track_dominant_object(boxes, prev_dominant, frame.shape)
            dominant_history.append(persistence)
            prev_dominant = curr_dominant
            
            timestamps.append(t)
            prev_gray = gray
            frame_idx += 1
            t += 1 / fps

        pbar.close()
        cap.release()

        # ---- Rhythm ----
        R = [self.rhythm_score(flow_signal[:i+1]) for i in range(len(flow_signal))]

        # ---- Attention Persistence Processing ----
        AP = self.ema_filter(dominant_history, alpha=0.3)
        # Invert: low persistence = attention shift = spike
        AP = 1 - AP

        # ---- Normalize base features ----
        M = self.robust_normalize(M)
        S = self.robust_normalize(S)
        C = self.robust_normalize(C)
        R = self.robust_normalize(R)
        I = self.robust_normalize(I)
        T = self.robust_normalize(T)
        SB = self.robust_normalize(SB)
        AP = self.robust_normalize(AP)

        # ---- Compute raw scores (without EV) ----
        raw_scores = np.array([
            0.20 * M[i] + 0.20 * S[i] + 0.12 * C[i] + 
            0.08 * R[i] + 0.08 * I[i] + 0.12 * T[i] + 
            0.05 * AP[i] + 0.05 * SB[i]
            for i in range(len(M))
        ])

        # ---- Expectation Violation (second derivative of energy) ----
        dE = np.diff(raw_scores, prepend=raw_scores[0])
        ddE = np.diff(dE, prepend=dE[0])
        EV = self.robust_normalize(np.abs(ddE))

        # ---- Final composite score ----
        scores = np.array([
            self.visual_score(M[i], S[i], C[i], R[i], I[i], T[i], EV[i], AP[i], SB[i])
            for i in range(len(M))
        ])

        timestamps = np.array(timestamps)

        np.savez(cache_file, timestamps=timestamps, scores=scores)
        print(f"[CACHE] Saved to {cache_file}")

        return timestamps, scores
    
    # ================== CLIP EXTRACTION ==================
    
    @staticmethod
    def extract_visual_clips(timestamps, scores, min_len=2, threshold=0.6, padding=1.5):
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
        window = gaussian(5, 1)
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
    
    def generate_clips(self, video_path, target_fps=1, min_len=2, threshold=0.6, use_cache=False):
        """
        Complete pipeline: analyze video and extract viral clips.
        
        Args:
            video_path: Path to video file
            target_fps: Sampling rate for analysis
            min_len: Minimum clip length in seconds
            threshold: Peak prominence threshold
            use_cache: Whether to use cached results
            
        Returns:
            timestamps: Array of timestamps
            scores: Array of excitement scores
            clips: List of (start, end) tuples for viral segments
        """
        timestamps, scores = self.compute_visual_scores(
            video_path, 
            target_fps=target_fps, 
            use_cache=use_cache
        )
        clips = self.extract_visual_clips(timestamps, scores, min_len=min_len, threshold=threshold)
        return timestamps, scores, clips
    
    # ================== VISUALIZATION ==================
    
    @staticmethod
    def plot_results(timestamps, scores, clips, title="Visual Excitement Timeline"):
        """
        Plot excitement scores and highlight viral clips.
        
        Args:
            timestamps: Array of timestamps
            scores: Array of excitement scores
            clips: List of (start, end) tuples
            title: Plot title
        """
        plt.figure(figsize=(14, 4))
        plt.plot(timestamps, scores, label="Visual Excitement Score", linewidth=1.5)
        for s, e in clips:
            plt.axvspan(s, e, color="red", alpha=0.3)
        plt.title(title)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_clips(clips):
        """Print detected clips in formatted output."""
        print("\n🎬 VIRAL CLIPS FOUND:")
        for i, (s, e) in enumerate(clips, 1):
            print(f"Clip {i}: Start: {s:.2f}s  End: {e:.2f}s  Duration: {e-s:.2f}s")


# ================== EXAMPLE USAGE ==================

if __name__ == "__main__":
    # Initialize detector
    detector = ClipVideo()
    
    # Process video
    video_path = r"D:\Clip_Extract\video1.mp4"
    timestamps, scores, clips = detector.generate_clips(
        video_path=video_path,
        target_fps=2,
        min_len=3,
        threshold=0.1,
        use_cache=False
    )
    
    # Display results
    detector.print_clips(clips)
    detector.plot_results(timestamps, scores, clips, title="Enhanced Visual Excitement Timeline (Red = Viral Clips)")