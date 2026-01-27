import mediapipe as mp
import cv2
import numpy as np
import os

class HandLandmarks:
    def __init__(self, model_path='models/hand_landmarker.task', num_hands=2, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a landmarker instance with the video mode:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.mp_drawing = mp.solutions.drawing_utils if hasattr(mp, 'solutions') else None # Fallback or custom draw
        self.timestamp_ms = 0

    def extract_landmarks(self, image):
        """
        Processes an image/frame and extracts hand landmarks.
        Input: image (BGR numpy array)
        Output: result object, list of landmark dicts
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # In VIDEO mode, we need to provide a timestamp
        self.timestamp_ms += 33 # approx 30fps
        
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        
        extracted_data = []
        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # hand_handedness is a list of categories
                handedness = detection_result.handedness[idx][0].category_name
                
                hand_info = {
                    'type': handedness,
                    'landmarks': []
                }
                for lm in hand_landmarks:
                    hand_info['landmarks'].append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                    })
                extracted_data.append(hand_info)
                
        return detection_result, extracted_data

    def draw_landmarks(self, image, detection_result):
        """
        Draws landmarks and connections on the image to create a mesh effect.
        """
        if not detection_result.hand_landmarks:
            return image
            
        # Standard MediaPipe Hand Connections
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),           # Index
            (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
            (5, 9), (9, 13), (13, 17), (0, 17)        # Palm
        ]

        h, w, _ = image.shape
        
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw Connections (Mesh lines)
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = hand_landmarks[start_idx]
                end_point = hand_landmarks[end_idx]
                
                px_start = (int(start_point.x * w), int(start_point.y * h))
                px_end = (int(end_point.x * w), int(end_point.y * h))
                
                cv2.line(image, px_start, px_end, (220, 220, 220), 2) # Light gray lines

            # Draw Landmarks (Dots)
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Outer white circle
                cv2.circle(image, (cx, cy), 6, (255, 255, 255), -1)
                # Inner color center (based on depth/z if visualized, but here just red)
                cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
             
        return image

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=1, help='Camera index (default: 1 for secondary cam)')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(current_dir), 'models', 'hand_landmarker.task')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        exit(1)

    print(f"Opening camera {args.camera} (looking for 'e2eSoft iVCam')...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}. Trying index 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             print("Error: Could not open any camera.")
             exit(1)

    detector = HandLandmarks(model_path=model_path)
    
    print("Press 'q' to quit...")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        results, data = detector.extract_landmarks(image)
        image = detector.draw_landmarks(image, results)
        
        # Add visual label
        cv2.putText(image, "Hand Landmark Preview", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Hands Preview', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
