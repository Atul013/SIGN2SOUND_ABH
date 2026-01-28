# Features Module

This module contains scripts for extracting features from video input, specifically focusing on landmarks for sign language recognition.

## `hand_landmarks.py`

This script provides the `HandLandmarks` class, which wraps MediaPipe Hands to extract 21 3D landmarks per hand.

### Usage

```python
from features.hand_landmarks import HandLandmarks
import cv2

detector = HandLandmarks(model_path='models/hand_landmarker.task')
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

## `pose_estimation.py`

Extracts upper body pose landmarks (shoulders, elbows, wrists, hips) for sign language context.

### Usage

```python
from features.pose_estimation import PoseEstimator

estimator = PoseEstimator()
results, pose_data = estimator.extract_pose(image)

# pose_data contains:
# - upper_body: dict of named landmarks
# - all_landmarks: list of all 33 pose landmarks
```

## `feature_utils.py`

Utilities for data contract, serialization, and normalization.

### Data Contract

```python
from features.feature_utils import FeatureContract

# Create standardized data
hand_data = FeatureContract.create_hand_data('Right', landmarks)
frame_data = FeatureContract.create_frame_data(0, [hand_data], pose_data)

# Validate
is_valid = FeatureContract.validate_frame_data(frame_data)
```

### Serialization

```python
from features.feature_utils import FeatureSerializer

# Convert to numpy
array = FeatureSerializer.to_numpy(landmarks)

# Save/load
FeatureSerializer.to_json(data, 'output.json')
data = FeatureSerializer.from_json('output.json')
```

### Normalization

```python
from features.feature_utils import FeatureNormalizer

# Normalize to wrist
normalized = FeatureNormalizer.normalize_hand_to_wrist(hand_data)

# Scale to unit box
scaled = FeatureNormalizer.scale_to_unit_box(landmarks)
```
