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
        Draws landmarks on the image. 
        Note: mp.solutions.drawing_utils might be missing in 3.13, so we might need manual drawing.
        """
        if not detection_result.hand_landmarks:
            return image
            
        # Manual drawing since mp.solutions might be missing
        for hand_landmarks in detection_result.hand_landmarks:
             for lm in hand_landmarks:
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                
             # simple connection drawing could be added here if needed
             # For now, just dots is enough verification
             
        return image

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust model path if run from features dir
    model_path = os.path.join(os.path.dirname(current_dir), 'models', 'hand_landmarker.task')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        exit(1)

    cap = cv2.VideoCapture(0)
    detector = HandLandmarks(model_path=model_path)
    
    print("Press 'q' to quit...")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        results, data = detector.extract_landmarks(image)
        image = detector.draw_landmarks(image, results)
        
        if data:
            print(f"Detected {len(data)} hand(s). Hand 1 Top Tip (y): {data[0]['landmarks'][8]['y']:.4f}", end='\r')

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
