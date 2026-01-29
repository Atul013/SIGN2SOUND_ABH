"""
Enhanced Real-time ASL Alphabet Recognition with Sentence Builder
Features:
- Improved hand detection with visual feedback
- Hand type display (Left/Right)
- Sentence building with minimal keys
- Text-to-speech output
- Better user experience
"""

import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path
import time
import pyttsx3
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.asl_model import create_model
from features.hand_landmarks import HandLandmarks
from data.vocabulary import ID_TO_LETTER, NUM_CLASSES


class EnhancedASLDemo:
    """Enhanced real-time ASL recognition with sentence building"""
    
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
        
        # Initialize hand landmark extractor with lower threshold for better detection
        print("Initializing MediaPipe Hand Landmarker...")
        self.extractor = HandLandmarks(
            model_path='models/hand_landmarker.task',
            num_hands=1,
            min_hand_detection_confidence=0.3,  # Lower threshold for better detection
            min_tracking_confidence=0.3
        )
        
        # Initialize text-to-speech
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_available = True
            print("Text-to-speech initialized")
        except:
            self.tts_available = False
            print("Text-to-speech not available")
        
        # Sentence building
        self.current_sentence = ""
        self.current_word = ""
        
        # Prediction smoothing with longer history
        self.prediction_history = deque(maxlen=10)
        self.stable_prediction = None
        self.stable_count = 0
        self.stability_threshold = 5  # Need 5 consistent predictions
        
        # Last added letter tracking
        self.last_added_letter = None
        self.last_add_time = 0
        self.add_cooldown = 1.5  # Seconds between auto-adds
        
        print("Initialization complete!\n")
    
    def preprocess_landmarks(self, landmarks):
        """Preprocess landmarks for model input"""
        # Manual wrist-relative normalization
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
        
        # Convert to numpy array
        features = np.array([[lm['x'], lm['y'], lm['z']] for lm in normalized_landmarks])
        features = features.flatten()
        
        # Convert to tensor
        features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        
        return features
    
    def predict(self, features):
        """Make prediction"""
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
    
    def update_stable_prediction(self, class_id, confidence):
        """Update stable prediction using history"""
        self.prediction_history.append(class_id)
        
        # Check if we have enough consistent predictions
        if len(self.prediction_history) >= self.stability_threshold:
            # Get most common prediction in recent history
            recent = list(self.prediction_history)[-self.stability_threshold:]
            most_common = max(set(recent), key=recent.count)
            count = recent.count(most_common)
            
            # If prediction is stable (appears in threshold number of frames)
            if count >= self.stability_threshold:
                if self.stable_prediction != most_common:
                    self.stable_prediction = most_common
                    self.stable_count = 0
                else:
                    self.stable_count += 1
            else:
                self.stable_count = 0
        
        return self.stable_prediction, self.stable_count
    
    def add_letter_to_word(self, letter):
        """Add letter to current word"""
        if letter and letter not in ['del', 'space', 'nothing']:
            current_time = time.time()
            # Prevent duplicate additions
            if letter != self.last_added_letter or (current_time - self.last_add_time) > self.add_cooldown:
                self.current_word += letter
                self.last_added_letter = letter
                self.last_add_time = current_time
                return True
        return False
    
    def speak_text(self, text):
        """Speak the given text"""
        if self.tts_available and text:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass
    
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
            cv2.line(image, start_point, end_point, (0, 255, 0), 3)
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark['x'] * w)
            y = int(landmark['y'] * h)
            cv2.circle(image, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(image, (x, y), 8, (255, 255, 255), 2)
        
        return image
    
    def draw_ui(self, frame, hand_detected, handedness, predicted_letter, confidence, stable_count):
        """Draw user interface elements"""
        h, w, _ = frame.shape
        
        # Create semi-transparent overlay for UI
        overlay = frame.copy()
        
        # Top bar background
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Bottom bar for sentence
        cv2.rectangle(frame, (0, h-150), (w, h), (40, 40, 40), -1)
        
        if hand_detected:
            # Display prediction with confidence
            pred_text = f"{predicted_letter}"
            conf_text = f"({confidence*100:.1f}%)"
            
            # Color based on stability
            if stable_count >= self.stability_threshold:
                color = (0, 255, 0)  # Green for stable
            elif stable_count >= 3:
                color = (0, 255, 255)  # Yellow for getting stable
            else:
                color = (0, 165, 255)  # Orange for unstable
            
            cv2.putText(frame, pred_text, (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
            cv2.putText(frame, conf_text, (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Display hand type
            hand_text = f"Hand: {handedness}"
            cv2.putText(frame, hand_text, (50, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Stability indicator
            stability_text = f"Stability: {'█' * stable_count}{'░' * (self.stability_threshold - stable_count)}"
            cv2.putText(frame, stability_text, (w - 400, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # No hand detected message
            cv2.putText(frame, "No hand detected", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, "Show your hand to camera", (50, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display current word and sentence
        word_text = f"Word: {self.current_word if self.current_word else '_'}"
        cv2.putText(frame, word_text, (30, h-100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        sentence_text = f"Sentence: {self.current_sentence if self.current_sentence else '_'}"
        cv2.putText(frame, sentence_text, (30, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame
    
    def run(self, camera_id=0):
        """Run enhanced demo"""
        print("="*80)
        print("Enhanced ASL Alphabet Recognition - Sentence Builder")
        print("="*80)
        print("\nControls:")
        print("  SPACE     - Add current letter to word")
        print("  ENTER     - Add word to sentence and speak it")
        print("  BACKSPACE - Delete last letter from word")
        print("  DELETE    - Clear current word")
        print("  C         - Clear entire sentence")
        print("  S         - Speak current sentence")
        print("  Q         - Quit")
        print("\nStarting webcam...\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            print("Make sure your webcam is connected and not in use by another application")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_time = time.time()
        fps = 0
        frame_count = 0
        
        print("Webcam opened successfully!")
        print("Window should appear now. If not, check your taskbar.\n")
        
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
            
            hand_detected = False
            handedness = "Unknown"
            predicted_letter = ""
            confidence = 0.0
            stable_count = 0
            
            # Process if hand detected
            if result and len(result) > 0:
                hand_detected = True
                hand_data = result[0]
                landmarks = hand_data['landmarks']
                
                # Get handedness (Left/Right)
                handedness = hand_data.get('handedness', 'Unknown')
                
                # Draw landmarks
                frame = self.draw_landmarks(frame, landmarks)
                
                # Preprocess and predict
                features = self.preprocess_landmarks(landmarks)
                class_id, confidence = self.predict(features)
                
                # Update stable prediction
                stable_pred, stable_count = self.update_stable_prediction(class_id, confidence)
                
                if stable_pred is not None:
                    predicted_letter = ID_TO_LETTER[stable_pred]
            else:
                # Reset prediction history when no hand
                self.prediction_history.clear()
                self.stable_prediction = None
                self.stable_count = 0
            
            # Draw UI
            frame = self.draw_ui(frame, hand_detected, handedness, predicted_letter, confidence, stable_count)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
            
            # Device and FPS info
            device_text = "GPU" if self.device.type == 'cuda' else "CPU"
            info_text = f"Device: {device_text} | FPS: {fps:.1f}"
            cv2.putText(frame, info_text, (frame.shape[1] - 350, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('ASL Sentence Builder', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - add letter
                if predicted_letter and stable_count >= self.stability_threshold:
                    if self.add_letter_to_word(predicted_letter):
                        print(f"Added: {predicted_letter} -> Word: {self.current_word}")
            elif key == 13:  # Enter - add word to sentence
                if self.current_word:
                    self.current_sentence += self.current_word + " "
                    print(f"Word added: {self.current_word}")
                    print(f"Sentence: {self.current_sentence}")
                    self.speak_text(self.current_word)
                    self.current_word = ""
                    self.last_added_letter = None
            elif key == 8:  # Backspace - delete last letter
                if self.current_word:
                    self.current_word = self.current_word[:-1]
                    print(f"Deleted letter -> Word: {self.current_word}")
                    self.last_added_letter = None
            elif key == ord('d') or key == 127:  # Delete - clear word
                self.current_word = ""
                self.last_added_letter = None
                print("Word cleared")
            elif key == ord('c'):  # C - clear sentence
                self.current_sentence = ""
                self.current_word = ""
                self.last_added_letter = None
                print("Sentence cleared")
            elif key == ord('s'):  # S - speak sentence
                if self.current_sentence:
                    print(f"Speaking: {self.current_sentence}")
                    self.speak_text(self.current_sentence)
            elif key == ord('p'):  # P - save screenshot
                screenshot_path = f"sentence_screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended.")
        if self.current_sentence:
            print(f"Final sentence: {self.current_sentence}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced ASL Sentence Builder')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    args = parser.parse_args()
    
    try:
        demo = EnhancedASLDemo(args.model, use_cuda=args.use_cuda)
        demo.run(camera_id=args.camera)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
