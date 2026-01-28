"""
Main preprocessing pipeline that integrates all components.
Converts raw video to segmented landmark sequences ready for model training/inference.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.hand_landmarks import HandLandmarks
from features.pose_estimation import PoseEstimator
from features.feature_utils import FeatureContract, FeatureSerializer, FeatureNormalizer
from preprocessing.temporal_segmentation import TemporalSegmenter
from preprocessing.robustness import LandmarkFilter, OcclusionHandler


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline from video to segmented sequences.
    """
    
    def __init__(self,
                 hand_model_path: str = 'models/hand_landmarker.task',
                 pose_model_path: Optional[str] = None,
                 enable_pose: bool = False,
                 enable_filtering: bool = True,
                 enable_occlusion_handling: bool = True):
        """
        Initialize preprocessing pipeline.
        
        Args:
            hand_model_path: Path to hand landmarker model
            pose_model_path: Path to pose landmarker model
            enable_pose: Whether to extract pose landmarks
            enable_filtering: Whether to apply noise filtering
            enable_occlusion_handling: Whether to handle occlusions
        """
        self.hand_detector = HandLandmarks(model_path=hand_model_path)
        
        self.pose_estimator = None
        if enable_pose:
            try:
                self.pose_estimator = PoseEstimator(model_path=pose_model_path)
            except Exception as e:
                print(f"Warning: Could not initialize pose estimator: {e}")
        
        self.segmenter = TemporalSegmenter()
        
        self.filter = LandmarkFilter() if enable_filtering else None
        self.occlusion_handler = OcclusionHandler() if enable_occlusion_handling else None
        
        self.frame_count = 0
    
    def process_frame(self, image: np.ndarray) -> Dict:
        """
        Process a single frame through the pipeline.
        
        Args:
            image: BGR image from video
            
        Returns:
            Frame data dict with hands and optional pose
        """
        # Extract hand landmarks
        _, hand_data = self.hand_detector.extract_landmarks(image)
        
        # Apply filtering if enabled
        if self.filter and hand_data:
            hand_data = self.filter.moving_average(hand_data)
        
        # Handle occlusions if enabled
        if self.occlusion_handler and hand_data:
            hand_data = self.occlusion_handler.process_frame(hand_data)
        
        # Extract pose if enabled
        pose_data = None
        if self.pose_estimator:
            _, pose_data_list = self.pose_estimator.extract_pose(image)
            if pose_data_list:
                pose_data = pose_data_list[0]
        
        # Create standardized frame data
        frame_data = FeatureContract.create_frame_data(
            frame_id=self.frame_count,
            hands=hand_data,
            pose=pose_data
        )
        
        self.frame_count += 1
        return frame_data
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> List[Dict]:
        """
        Process entire video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            
        Returns:
            List of frame data dicts
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            
            frame_data = self.process_frame(image)
            frames.append(frame_data)
            
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
        
        cap.release()
        return frames
    
    def segment_video(self, video_path: str, output_dir: str = 'data/processed') -> List[Dict]:
        """
        Process and segment video into individual signs.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save processed segments
            
        Returns:
            List of segmented sign dicts
        """
        # Process all frames
        print(f"Processing video: {video_path}")
        frames = self.process_video(video_path)
        print(f"Extracted {len(frames)} frames")
        
        # Extract just hand landmarks for segmentation
        hand_sequence = [frame['hands'] for frame in frames]
        
        # Segment into signs
        print("Segmenting signs...")
        segments = self.segmenter.segment_sequence(hand_sequence)
        print(f"Detected {len(segments)} sign(s)")
        
        # Save segments
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, segment in enumerate(segments):
            # Get full frame data for this segment
            start, end = segment['start_frame'], segment['end_frame']
            segment_frames = frames[start:end+1]
            
            # Save as JSON
            output_path = os.path.join(output_dir, f'segment_{idx:03d}.json')
            FeatureSerializer.to_json({
                'segment_id': idx,
                'start_frame': start,
                'end_frame': end,
                'duration': segment['duration'],
                'frames': segment_frames
            }, output_path)
            
            print(f"  Saved segment {idx}: frames {start}-{end} to {output_path}")
        
        return segments
    
    def reset(self):
        """Reset pipeline state."""
        self.frame_count = 0
        if self.filter:
            self.filter.reset()
        if self.occlusion_handler:
            self.occlusion_handler.reset()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess video for sign language recognition')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--enable-pose', action='store_true', help='Enable pose estimation')
    parser.add_argument('--test', action='store_true', help='Run test with webcam')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(enable_pose=args.enable_pose)
    
    if args.test:
        # Test with webcam
        print("Testing preprocessing pipeline with webcam...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        frames = []
        
        while cap.isOpened() and len(frames) < 100:
            ret, image = cap.read()
            if not ret:
                break
            
            frame_data = pipeline.process_frame(image)
            frames.append(frame_data)
            
            # Display
            cv2.putText(image, f"Frames: {len(frames)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Preprocessing Test', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {len(frames)} frames")
        
        # Try segmentation
        if len(frames) > 10:
            hand_sequence = [frame['hands'] for frame in frames]
            segments = pipeline.segmenter.segment_sequence(hand_sequence)
            print(f"Detected {len(segments)} sign(s)")
    
    elif args.video:
        # Process video file
        segments = pipeline.segment_video(args.video, args.output)
        print(f"\nProcessing complete! Saved {len(segments)} segments to {args.output}")
    
    else:
        print("Please specify --video or --test")
        parser.print_help()
