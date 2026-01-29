"""
Real-time ASL Alphabet Recognition Demo
Uses webcam to detect and classify ASL hand signs in real-time
"""

import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.asl_model import create_model
from features.hand_landmarks import HandLandmarks
from features.feature_utils import FeatureNormalizer
from data.vocabulary import ID_TO_LETTER, NUM_CLASSES


class ASLRealtimeDemo:
    """Real-time ASL recognition demo"""
    
    def __init__(self, model_path, use_cuda=True):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model = create_model(
            model_type='gru',
            input_dim=63,
            num_classes=NUM_CLASSES,
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded (Val Acc: {checkpoint['val_accuracy']:.4f})")
        
        # Initialize hand landmark extractor
        print("Initializing MediaPipe Hand Landmarker...")
        self.extractor = HandLandmarks(
            model_path='models/hand_landmarker.task',
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Prediction smoothing
        self.prediction_history = []
        self.history_size = 5
        
        print("Initialization complete!\n")
    
    def preprocess_landmarks(self, landmarks):
        """Preprocess landmarks for model input"""
        # Manual wrist-relative normalization
        # Get wrist position (landmark 0)
        wrist = landmarks[0]
        wrist_x, wrist_y, wrist_z = wrist['x'], wrist['y'], wrist['z']
        
        # Normalize all landmarks relative to wrist
        normalized_landmarks = []
        for lm in landmarks:
            normalized_landmarks.append({
                'x': lm['x'] - wrist_x,
                'y': lm['y'] - wrist_y,
                'z': lm['z'] - wrist_z
            })
        
        # Convert to numpy array (21 landmarks * 3 coordinates = 63 features)
        features = np.array([[lm['x'], lm['y'], lm['z']] for lm in normalized_landmarks])
        features = features.flatten()
        
        # Convert to tensor
        features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # (1, 1, 63)
        
        return features
    
    def predict(self, features):
        """Make prediction"""
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
    
    def smooth_prediction(self, class_id):
        """Smooth predictions using history"""
        self.prediction_history.append(class_id)
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Return most common prediction
        if len(self.prediction_history) >= 3:
            return max(set(self.prediction_history), key=self.prediction_history.count)
        return class_id
    
    def draw_landmarks(self, image, landmarks):
        """Draw hand landmarks on image"""
        h, w, _ = image.shape
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            start_point = (int(landmarks[start_idx]['x'] * w), 
                          int(landmarks[start_idx]['y'] * h))
            end_point = (int(landmarks[end_idx]['x'] * w), 
                        int(landmarks[end_idx]['y'] * h))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark['x'] * w)
            y = int(landmark['y'] * h)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        
        return image
    
    def run(self, camera_id=0):
        """Run real-time demo"""
        print("="*70)
        print("ASL Alphabet Recognition - Real-Time Demo")
        print("="*70)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("\nStarting webcam...\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps_time = time.time()
        fps = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract landmarks
            detection_result, result = self.extractor.extract_landmarks(rgb_frame)
            
            # Process if hand detected
            if result and len(result) > 0:
                hand_data = result[0]
                landmarks = hand_data['landmarks']
                
                # Draw landmarks
                frame = self.draw_landmarks(frame, landmarks)
                
                # Preprocess and predict
                features = self.preprocess_landmarks(landmarks)
                class_id, confidence = self.predict(features)
                
                # Smooth prediction
                smoothed_class_id = self.smooth_prediction(class_id)
                predicted_letter = ID_TO_LETTER[smoothed_class_id]
                
                # Display prediction
                text = f"{predicted_letter} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                
                # Display handedness
                handedness = hand_data.get('handedness', 'Unknown')
                cv2.putText(frame, f"Hand: {handedness}", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # No hand detected
                cv2.putText(frame, "No hand detected", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Display device info
            device_text = "GPU" if self.device.type == 'cuda' else "CPU"
            cv2.putText(frame, f"Device: {device_text}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('ASL Alphabet Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Real-Time Recognition Demo')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    args = parser.parse_args()
    
    try:
        demo = ASLRealtimeDemo(args.model, use_cuda=args.use_cuda)
        demo.run(camera_id=args.camera)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
