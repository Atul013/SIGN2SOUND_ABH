"""
Feature extraction from raw video frames.
Extracts frames, applies preprocessing, and prepares data for landmark extraction.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
from pathlib import Path


class VideoFrameExtractor:
    """
    Extracts and preprocesses frames from video files.
    """
    
    def __init__(self, 
                 target_fps: Optional[int] = None,
                 resize_dimensions: Optional[Tuple[int, int]] = None,
                 normalize: bool = True):
        """
        Initialize the frame extractor.
        
        Args:
            target_fps: Target frame rate (None = use original)
            resize_dimensions: Target (width, height) for resizing (None = no resize)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_fps = target_fps
        self.resize_dimensions = resize_dimensions
        self.normalize = normalize
    
    def extract_frames(self, video_path: str, 
                      max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (None = all)
            
        Returns:
            Tuple of (frames list, metadata dict)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video metadata
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame sampling rate
        if self.target_fps and self.target_fps < original_fps:
            frame_skip = int(original_fps / self.target_fps)
        else:
            frame_skip = 1
        
        frames = []
        frame_indices = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames based on target FPS
            if frame_count % frame_skip == 0:
                processed_frame = self._preprocess_frame(frame)
                frames.append(processed_frame)
                frame_indices.append(frame_count)
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        metadata = {
            'video_path': video_path,
            'original_fps': original_fps,
            'target_fps': self.target_fps or original_fps,
            'total_frames': total_frames,
            'extracted_frames': len(frames),
            'frame_indices': frame_indices,
            'original_dimensions': (width, height),
            'processed_dimensions': self.resize_dimensions or (width, height)
        }
        
        return frames, metadata
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Preprocessed frame
        """
        # Resize if needed
        if self.resize_dimensions:
            frame = cv2.resize(frame, self.resize_dimensions)
        
        # Normalize if needed
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def save_frames(self, frames: List[np.ndarray], 
                   output_dir: str, 
                   prefix: str = "frame") -> List[str]:
        """
        Save extracted frames to disk.
        
        Args:
            frames: List of frame arrays
            output_dir: Directory to save frames
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i, frame in enumerate(frames):
            # Denormalize if needed
            if self.normalize:
                frame = (frame * 255).astype(np.uint8)
            
            filename = f"{prefix}_{i:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)
        
        return saved_paths


class FeatureExtractor:
    """
    Extracts features from video for landmark detection pipeline.
    Combines frame extraction with quality checks and preprocessing.
    """
    
    def __init__(self, 
                 target_fps: int = 30,
                 min_brightness: float = 30.0,
                 max_blur_threshold: float = 100.0):
        """
        Initialize feature extractor.
        
        Args:
            target_fps: Target frame rate for extraction
            min_brightness: Minimum average brightness (0-255)
            max_blur_threshold: Maximum blur variance threshold
        """
        self.frame_extractor = VideoFrameExtractor(target_fps=target_fps)
        self.min_brightness = min_brightness
        self.max_blur_threshold = max_blur_threshold
    
    def check_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Check quality metrics for a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            Dict with quality metrics
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness (average pixel value)
        brightness = np.mean(gray)
        
        # Blur detection (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        return {
            'brightness': brightness,
            'blur_score': blur_score,
            'contrast': contrast,
            'is_acceptable': (brightness >= self.min_brightness and 
                            blur_score >= self.max_blur_threshold)
        }
    
    def extract_with_quality_check(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract frames with quality filtering.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (quality frames, metadata)
        """
        frames, metadata = self.frame_extractor.extract_frames(video_path)
        
        quality_frames = []
        quality_metrics = []
        rejected_count = 0
        
        for frame in frames:
            # Denormalize for quality check
            check_frame = (frame * 255).astype(np.uint8) if self.frame_extractor.normalize else frame
            
            quality = self.check_frame_quality(check_frame)
            quality_metrics.append(quality)
            
            if quality['is_acceptable']:
                quality_frames.append(frame)
            else:
                rejected_count += 1
        
        metadata['quality_metrics'] = {
            'total_frames': len(frames),
            'accepted_frames': len(quality_frames),
            'rejected_frames': rejected_count,
            'average_brightness': np.mean([q['brightness'] for q in quality_metrics]),
            'average_blur_score': np.mean([q['blur_score'] for q in quality_metrics]),
            'average_contrast': np.mean([q['contrast'] for q in quality_metrics])
        }
        
        return quality_frames, metadata
    
    def batch_extract(self, video_paths: List[str], 
                     output_dir: Optional[str] = None) -> Dict[str, Tuple[List[np.ndarray], Dict]]:
        """
        Extract features from multiple videos.
        
        Args:
            video_paths: List of video file paths
            output_dir: Optional directory to save extracted frames
            
        Returns:
            Dict mapping video paths to (frames, metadata) tuples
        """
        results = {}
        
        for video_path in video_paths:
            print(f"Processing: {video_path}")
            
            try:
                frames, metadata = self.extract_with_quality_check(video_path)
                results[video_path] = (frames, metadata)
                
                # Save frames if output directory specified
                if output_dir:
                    video_name = Path(video_path).stem
                    frame_dir = os.path.join(output_dir, video_name)
                    self.frame_extractor.save_frames(frames, frame_dir, prefix=video_name)
                
                print(f"  ✓ Extracted {len(frames)} frames")
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                results[video_path] = ([], {'error': str(e)})
        
        return results


class TemporalFeatureExtractor:
    """
    Extracts temporal features from frame sequences.
    Computes motion, optical flow, and temporal statistics.
    """
    
    @staticmethod
    def compute_optical_flow(prev_frame: np.ndarray, 
                            curr_frame: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between consecutive frames.
        
        Args:
            prev_frame: Previous frame (BGR)
            curr_frame: Current frame (BGR)
            
        Returns:
            Optical flow field
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        return flow
    
    @staticmethod
    def compute_motion_magnitude(frames: List[np.ndarray]) -> List[float]:
        """
        Compute motion magnitude between consecutive frames.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            List of motion magnitudes
        """
        if len(frames) < 2:
            return [0.0]
        
        magnitudes = []
        
        for i in range(1, len(frames)):
            flow = TemporalFeatureExtractor.compute_optical_flow(frames[i-1], frames[i])
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_magnitude = np.mean(magnitude)
            magnitudes.append(avg_magnitude)
        
        return magnitudes
    
    @staticmethod
    def compute_temporal_statistics(frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute temporal statistics for a sequence.
        
        Args:
            frames: List of frames
            
        Returns:
            Dict with temporal statistics
        """
        motion_magnitudes = TemporalFeatureExtractor.compute_motion_magnitude(frames)
        
        return {
            'sequence_length': len(frames),
            'mean_motion': np.mean(motion_magnitudes),
            'max_motion': np.max(motion_magnitudes),
            'motion_variance': np.var(motion_magnitudes),
            'total_motion': np.sum(motion_magnitudes)
        }


if __name__ == "__main__":
    # Test feature extraction
    print("Testing Feature Extraction...")
    
    # This is a placeholder test - requires actual video file
    print("\n⚠️  Note: Feature extraction requires video files to test.")
    print("Example usage:")
    print("  extractor = FeatureExtractor(target_fps=30)")
    print("  frames, metadata = extractor.extract_with_quality_check('path/to/video.mp4')")
    print("  print(f'Extracted {len(frames)} frames')")
    print("  print(f'Metadata: {metadata}')")
    
    print("\n✅ Feature extraction module ready!")
