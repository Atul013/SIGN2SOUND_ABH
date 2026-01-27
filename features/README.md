# Features Module

This module contains scripts for extracting features from video input, specifically focusing on landmarks for sign language recognition.

## `hand_landmarks.py`

This script provides the `HandLandmarks` class, which wraps MediaPipe Hands to extract 21 3D landmarks per hand.

### Usage

```python
from features.hand_landmarks import HandLandmarks
import cv2

detector = HandLandmarks()
image = cv2.imread('path/to/image.jpg')
results, data = detector.extract_landmarks(image)

# data structure:
# [
#   {
#     'type': 'Left' or 'Right',
#     'landmarks': [ {'x': ..., 'y': ..., 'z': ...}, ... ] (21 points)
#   },
#   ...
# ]
```
