"""
Real-Time ASL Alphabet Recognition Demo

This script demonstrates real-time ASL alphabet recognition using webcam input.
It integrates Developer A's temporal pipeline with Developer B's model and TTS.
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.infer import ASLInference
from inference.tts import TextToSpeech
from data.vocabulary import ID_TO_LETTER

# Import Developer A's modules
try:
    from features.hand_landmarks import extract_hand_landmarks
    from preprocessing.robustness import apply_temporal_smoothing
    DEVELOPER_A_AVAILABLE = True
except ImportError:
    DEVELOPER_A_AVAILABLE = False
    print("⚠️  Developer A's modules not available")


class RealTimeASLDemo:
    """
    Real-time ASL alphabet recognition demo.
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/best_model.pth",
        camera_id: int = 0,
        confidence_threshold: float = 0.7,
        enable_tts: bool = True,
        display_window: bool = True
    ):
        """
        Initialize real-time demo.
        
        Args:
            model_path: Path to trained model
            camera_id: Camera device ID
            confidence_threshold: Minimum confidence for predictions
            enable_tts: Whether to enable text-to-speech
            display_window: Whether to display video window
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.enable_tts = enable_tts
        self.display_window = display_window
        
        # Initialize inference engine
        print("Initializing inference engine...")
        self.inference = ASLInference(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            temporal_smoothing=True
        )
        
        # Initialize TTS
        if self.enable_tts:
            print("Initializing text-to-speech...")
            self.tts = TextToSpeech(engine='pyttsx3', rate=150)
        else:
            self.tts = None
        
        # Initialize camera
        print(f"Initializing camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # State tracking
        self.current_word = []
        self.last_letter = None
        self.last_letter_time = 0
        self.letter_hold_time = 1.0  # Seconds to hold before adding to word
        self.word_pause_time = 2.0  # Seconds of no detection to complete word
        
        # Frame buffer for temporal processing
        self.frame_buffer = deque(maxlen=30)
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        print("✅ Real-time demo initialized!")
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from video frame.
        
        Args:
            frame: Video frame (BGR image)
            
        Returns:
            Feature array of shape (1, feature_dim)
        """
        if DEVELOPER_A_AVAILABLE:
            # Use Developer A's landmark extraction
            landmarks = extract_hand_landmarks(frame)
            
            # Convert to feature vector
            features = self._landmarks_to_features(landmarks)
        else:
            # Simulate feature extraction
            features = np.random.rand(63)
        
        return features
    
    def _landmarks_to_features(self, landmarks: dict) -> np.ndarray:
        """Convert landmarks to feature vector."""
        features = []
        
        if landmarks and 'hand_landmarks' in landmarks:
            hand_lms = landmarks['hand_landmarks']
            
            for i in range(21):
                if i in hand_lms:
                    lm = hand_lms[i]
                    features.extend([lm['x'], lm['y'], lm['z']])
                else:
                    features.extend([0.0, 0.0, 0.0])
        else:
            # No hand detected
            features = [0.0] * 63
        
        return np.array(features, dtype=np.float32)
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame.
        
        Args:
            frame: Video frame
            
        Returns:
            Tuple of (predicted_letter, confidence, annotated_frame)
        """
        # Extract features
        features = self.extract_features(frame)
        
        # Add to buffer
        self.frame_buffer.append(features)
        
        # Need enough frames for prediction
        if len(self.frame_buffer) < 10:
            return None, 0.0, frame
        
        # Convert buffer to sequence
        feature_sequence = np.array(list(self.frame_buffer))
        
        # Run inference
        letter, confidence = self.inference.predict_letter(
            feature_sequence,
            return_confidence=True
        )
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame, letter, confidence)
        
        return letter, confidence, annotated_frame
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        letter: str,
        confidence: float
    ) -> np.ndarray:
        """
        Annotate frame with predictions and info.
        
        Args:
            frame: Video frame
            letter: Predicted letter
            confidence: Prediction confidence
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw prediction
        if letter:
            # Large letter display
            cv2.putText(
                annotated,
                letter,
                (w // 2 - 50, h // 2),
                cv2.FONT_HERSHEY_BOLD,
                5,
                (0, 255, 0),
                10
            )
            
            # Confidence bar
            bar_width = int(300 * confidence)
            cv2.rectangle(
                annotated,
                (20, h - 60),
                (20 + bar_width, h - 40),
                (0, 255, 0),
                -1
            )
            cv2.rectangle(
                annotated,
                (20, h - 60),
                (320, h - 40),
                (255, 255, 255),
                2
            )
            
            # Confidence text
            cv2.putText(
                annotated,
                f"Confidence: {confidence:.2f}",
                (20, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Draw current word
        word_text = ''.join(self.current_word) if self.current_word else "[No word]"
        cv2.putText(
            annotated,
            f"Word: {word_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Draw FPS
        cv2.putText(
            annotated,
            f"FPS: {self.fps:.1f}",
            (w - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw instructions
        instructions = [
            "Hold letter for 1s to add to word",
            "Pause 2s to speak word",
            "Press 'c' to clear word",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(
                annotated,
                instruction,
                (20, h - 150 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
        
        return annotated
    
    def update_word(self, letter: str, confidence: float):
        """
        Update current word based on detected letter.
        
        Args:
            letter: Detected letter
            confidence: Detection confidence
        """
        current_time = time.time()
        
        if letter and confidence >= self.confidence_threshold:
            # Same letter as before
            if letter == self.last_letter:
                # Check if held long enough
                if current_time - self.last_letter_time >= self.letter_hold_time:
                    # Add to word (only once)
                    if not self.current_word or self.current_word[-1] != letter:
                        self.current_word.append(letter)
                        print(f"Added letter: {letter}")
                        
                        if self.tts:
                            self.tts.speak_letter(letter)
            else:
                # New letter
                self.last_letter = letter
                self.last_letter_time = current_time
        else:
            # No detection or low confidence
            if self.last_letter and current_time - self.last_letter_time >= self.word_pause_time:
                # Pause detected - speak word
                if self.current_word:
                    word = ''.join(self.current_word)
                    print(f"\n🔊 Speaking word: {word}")
                    
                    if self.tts:
                        self.tts.speak_word(word)
                    
                    # Clear word
                    self.current_word = []
                
                self.last_letter = None
    
    def run(self):
        """Run the real-time demo."""
        print("\n" + "=" * 60)
        print("Starting Real-Time ASL Alphabet Recognition")
        print("=" * 60)
        print("\nInstructions:")
        print("  - Hold a letter sign for 1 second to add it to the word")
        print("  - Pause for 2 seconds to speak the word")
        print("  - Press 'c' to clear the current word")
        print("  - Press 'q' to quit")
        print("\n" + "=" * 60 + "\n")
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                letter, confidence, annotated_frame = self.process_frame(frame)
                
                # Update word
                self.update_word(letter, confidence)
                
                # Update FPS
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                
                # Display frame
                if self.display_window:
                    cv2.imshow('ASL Alphabet Recognition', annotated_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('c'):
                        print("\nClearing word...")
                        self.current_word = []
                        self.last_letter = None
        
        finally:
            # Cleanup
            self.cap.release()
            if self.display_window:
                cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print("Demo ended")
            print("=" * 60)


def main():
    """Main function."""
    print("ASL Alphabet Recognition - Real-Time Demo")
    print("=" * 70)
    
    try:
        # Create demo
        demo = RealTimeASLDemo(
            model_path="checkpoints/best_model.pth",
            camera_id=0,
            confidence_threshold=0.7,
            enable_tts=True,
            display_window=True
        )
        
        # Run demo
        demo.run()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: This demo requires:")
        print("1. Trained model in checkpoints/")
        print("2. Webcam access")
        print("3. OpenCV: pip install opencv-python")
        print("4. TTS engine: pip install pyttsx3")


if __name__ == "__main__":
    main()
