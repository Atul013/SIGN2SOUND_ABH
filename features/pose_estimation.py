import mediapipe as mp
import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple

class PoseEstimator:
    """
    Extracts upper body pose landmarks using MediaPipe Pose.
    Focuses on shoulders, elbows, and wrists for sign language context.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose estimator.
        
        Args:
            model_path: Path to pose landmarker model (if using tasks API)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        # Try to use tasks API first (for Python 3.13 compatibility)
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            # Use default model if not provided
            if model_path is None:
                model_path = 'models/pose_landmarker.task'
            
            if os.path.exists(model_path):
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.VIDEO,
                    min_pose_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.landmarker = PoseLandmarker.create_from_options(options)
                self.use_tasks_api = True
                self.timestamp_ms = 0
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
                
        except (AttributeError, FileNotFoundError):
            # Fallback to solutions API if available
            if hasattr(mp, 'solutions'):
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.use_tasks_api = False
            else:
                raise RuntimeError("MediaPipe pose not available")
        
        # Upper body landmark indices (MediaPipe Pose has 33 landmarks)
        self.upper_body_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24
        }
    
    def extract_pose(self, image: np.ndarray) -> Tuple[Optional[object], List[Dict]]:
        """
        Extract pose landmarks from image.
        
        Args:
            image: BGR image from camera/video
            
        Returns:
            (results object, extracted pose data)
        """
        if self.use_tasks_api:
            return self._extract_with_tasks(image)
        else:
            return self._extract_with_solutions(image)
    
    def _extract_with_tasks(self, image: np.ndarray) -> Tuple[Optional[object], List[Dict]]:
        """Extract using tasks API."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        self.timestamp_ms += 33  # ~30fps
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        
        extracted_data = []
        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                pose_data = {
                    'upper_body': {},
                    'all_landmarks': []
                }
                
                # Extract all landmarks
                for idx, lm in enumerate(pose_landmarks):
                    pose_data['all_landmarks'].append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                    })
                
                # Extract upper body specifically
                for name, idx in self.upper_body_indices.items():
                    if idx < len(pose_landmarks):
                        lm = pose_landmarks[idx]
                        pose_data['upper_body'][name] = {
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z,
                            'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                        }
                
                extracted_data.append(pose_data)
        
        return detection_result, extracted_data
    
    def _extract_with_solutions(self, image: np.ndarray) -> Tuple[Optional[object], List[Dict]]:
        """Extract using solutions API."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        extracted_data = []
        if results.pose_landmarks:
            pose_data = {
                'upper_body': {},
                'all_landmarks': []
            }
            
            # Extract all landmarks
            for lm in results.pose_landmarks.landmark:
                pose_data['all_landmarks'].append({
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                })
            
            # Extract upper body specifically
            for name, idx in self.upper_body_indices.items():
                lm = results.pose_landmarks.landmark[idx]
                pose_data['upper_body'][name] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                }
            
            extracted_data.append(pose_data)
        
        return results, extracted_data
    
    def normalize_to_shoulders(self, pose_data: Dict) -> Dict:
        """
        Normalize pose landmarks relative to shoulder center.
        
        Args:
            pose_data: Extracted pose data
            
        Returns:
            Normalized pose data
        """
        if 'upper_body' not in pose_data:
            return pose_data
        
        upper_body = pose_data['upper_body']
        
        # Calculate shoulder center
        if 'left_shoulder' in upper_body and 'right_shoulder' in upper_body:
            left_sh = upper_body['left_shoulder']
            right_sh = upper_body['right_shoulder']
            
            center_x = (left_sh['x'] + right_sh['x']) / 2
            center_y = (left_sh['y'] + right_sh['y']) / 2
            center_z = (left_sh['z'] + right_sh['z']) / 2
            
            # Normalize all upper body landmarks
            normalized = {}
            for name, lm in upper_body.items():
                normalized[name] = {
                    'x': lm['x'] - center_x,
                    'y': lm['y'] - center_y,
                    'z': lm['z'] - center_z,
                    'visibility': lm.get('visibility', 1.0)
                }
            
            return {
                'upper_body': normalized,
                'all_landmarks': pose_data.get('all_landmarks', [])
            }
        
        return pose_data
    
    def draw_pose(self, image: np.ndarray, results: object) -> np.ndarray:
        """
        Draw pose landmarks on image.
        
        Args:
            image: Image to draw on
            results: Detection results
            
        Returns:
            Image with pose drawn
        """
        if self.use_tasks_api:
            return self._draw_with_tasks(image, results)
        else:
            return self._draw_with_solutions(image, results)
    
    def _draw_with_tasks(self, image: np.ndarray, results: object) -> np.ndarray:
        """Draw using tasks API (manual drawing)."""
        if not results.pose_landmarks:
            return image
        
        h, w, _ = image.shape
        
        # Define connections for upper body
        connections = [
            (11, 12),  # Shoulders
            (11, 13),  # Left shoulder to elbow
            (13, 15),  # Left elbow to wrist
            (12, 14),  # Right shoulder to elbow
            (14, 16),  # Right elbow to wrist
            (11, 23),  # Left shoulder to hip
            (12, 24),  # Right shoulder to hip
            (23, 24),  # Hips
        ]
        
        for pose_landmarks in results.pose_landmarks:
            # Draw connections
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                    start_lm = pose_landmarks[start_idx]
                    end_lm = pose_landmarks[end_idx]
                    
                    start_point = (int(start_lm.x * w), int(start_lm.y * h))
                    end_point = (int(end_lm.x * w), int(end_lm.y * h))
                    
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
            
            # Draw landmarks
            for idx in self.upper_body_indices.values():
                if idx < len(pose_landmarks):
                    lm = pose_landmarks[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        
        return image
    
    def _draw_with_solutions(self, image: np.ndarray, results: object) -> np.ndarray:
        """Draw using solutions API."""
        if hasattr(mp, 'solutions') and results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
        return image


if __name__ == "__main__":
    print("Testing Pose Estimation...")
    print("Note: This test requires a pose model file or will use solutions API")
    
    # Try to create estimator
    try:
        estimator = PoseEstimator()
        print(f"Using Tasks API: {estimator.use_tasks_api}")
        
        # Test with camera if available
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Press 'q' to quit...")
            for _ in range(30):  # Test for 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                results, pose_data = estimator.extract_pose(frame)
                if pose_data:
                    print(f"Detected pose with {len(pose_data[0]['upper_body'])} upper body landmarks")
                    break
            
            cap.release()
        else:
            print("No camera available for testing")
    
    except Exception as e:
        print(f"Pose estimation test skipped: {e}")
        print("This is expected if pose model is not downloaded")
