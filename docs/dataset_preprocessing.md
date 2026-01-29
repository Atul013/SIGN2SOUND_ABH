# Dataset Preprocessing Details

## Overview

This document describes the complete data preprocessing pipeline for the Sign2Sound Phase 2 project. The pipeline transforms raw video data into normalized landmark sequences suitable for temporal sign recognition.

---

## Pipeline Architecture

```
Raw Video Input
    ↓
Frame Extraction & Quality Check
    ↓
Landmark Detection (MediaPipe)
    ↓
Normalization & Filtering
    ↓
Temporal Segmentation
    ↓
Data Augmentation (Training Only)
    ↓
Serialized Landmark Sequences
```

---

## Step 1: Frame Extraction

### Purpose
Extract frames from video files at a consistent frame rate for landmark detection.

### Implementation
- **Module**: `preprocessing/extract_features.py`
- **Target FPS**: 30 fps (configurable)
- **Frame Sampling**: Uniform sampling to match target FPS
- **Quality Checks**:
  - Brightness: Minimum average brightness of 30/255
  - Blur Detection: Laplacian variance threshold of 100
  - Contrast: Standard deviation check

### Configuration
```python
extractor = FeatureExtractor(
    target_fps=30,
    min_brightness=30.0,
    max_blur_threshold=100.0
)
```

### Output
- Extracted frames (BGR format)
- Metadata: FPS, resolution, frame indices
- Quality metrics per frame

---

## Step 2: Landmark Detection

### Purpose
Extract hand and pose landmarks from video frames using MediaPipe.

### Implementation
- **Module**: `features/hand_landmarks.py`, `features/pose_estimation.py`
- **Hand Landmarks**: 21 keypoints per hand (x, y, z coordinates)
- **Pose Landmarks**: Upper body keypoints (shoulders, elbows, wrists)
- **Detection Confidence**: Minimum 0.5 for hand detection

### Hand Landmark Structure
```
Landmark Indices:
0: Wrist
1-4: Thumb (CMC, MCP, IP, Tip)
5-8: Index finger (MCP, PIP, DIP, Tip)
9-12: Middle finger (MCP, PIP, DIP, Tip)
13-16: Ring finger (MCP, PIP, DIP, Tip)
17-20: Pinky (MCP, PIP, DIP, Tip)
```

### Output Format
```python
{
    'type': 'Right' | 'Left',
    'landmarks': [
        {'x': float, 'y': float, 'z': float},  # 21 landmarks
        ...
    ],
    'confidence': float
}
```

---

## Step 3: Normalization

### Purpose
Normalize landmarks to be invariant to hand position, scale, and orientation.

### Normalization Strategies

#### 3.1 Wrist-Relative Normalization
- **Purpose**: Position invariance
- **Method**: Translate all landmarks relative to wrist (landmark 0)
- **Formula**: `landmark_norm = landmark - wrist`

#### 3.2 Scale Normalization
- **Purpose**: Size invariance
- **Method**: Normalize by hand bounding box diagonal
- **Formula**: `landmark_scaled = landmark_norm / bbox_diagonal`

#### 3.3 Coordinate Range Normalization
- **Purpose**: Consistent input range
- **Method**: Map coordinates to [0, 1] or [-1, 1]

### Implementation
```python
from features.feature_utils import FeatureNormalizer

normalizer = FeatureNormalizer(
    method='wrist_relative',
    scale_invariant=True,
    coordinate_range=(-1, 1)
)

normalized_landmarks = normalizer.normalize(landmarks)
```

---

## Step 4: Temporal Filtering

### Purpose
Remove noise and smooth landmark trajectories over time.

### Filtering Techniques

#### 4.1 Moving Average Filter
- **Window Size**: 3-5 frames
- **Purpose**: Reduce jitter in landmark positions

#### 4.2 Velocity Filtering
- **Purpose**: Remove unrealistic rapid movements
- **Threshold**: Maximum velocity based on frame rate

#### 4.3 Occlusion Handling
- **Detection**: Low confidence scores (\< 0.5)
- **Strategy**: Linear interpolation for short gaps (\< 5 frames)
- **Rejection**: Discard sequences with \> 30% missing data

### Implementation
```python
from preprocessing.robustness import LandmarkFilter

filter = LandmarkFilter(
    window_size=3,
    velocity_threshold=0.5,
    interpolate_missing=True
)

filtered_sequence = filter.apply(landmark_sequence)
```

---

## Step 5: Temporal Segmentation

### Purpose
Segment continuous video into individual sign sequences.

### Segmentation Algorithm

#### 5.1 Motion-Based Segmentation
1. **Compute Velocity**: Frame-to-frame landmark displacement
2. **Detect Motion Peaks**: Identify periods of high motion
3. **Identify Boundaries**: Low-motion periods indicate sign boundaries
4. **Validate Duration**: Minimum sign duration (10 frames)

#### 5.2 Pause Detection
- **Pause Threshold**: Velocity \< 0.005
- **Pause Duration**: Minimum 5 consecutive low-motion frames
- **Purpose**: Identify phrase boundaries

### Configuration
```python
from preprocessing.temporal_segmentation import TemporalSegmenter

segmenter = TemporalSegmenter(
    motion_threshold=0.02,
    min_sign_duration=10,
    pause_duration=5,
    pause_threshold=0.005
)

segments = segmenter.segment_sequence(landmark_sequence)
```

### Output Format
```python
{
    'start_frame': int,
    'end_frame': int,
    'duration': int,
    'landmarks': List[Dict]  # Landmark sequence
}
```

---

## Step 6: Data Augmentation (Training Only)

### Purpose
Increase training data diversity and improve model generalization.

### Augmentation Techniques

#### 6.1 Spatial Augmentations
- **Rotation**: ±15 degrees around wrist center
- **Scaling**: 0.9x to 1.1x
- **Translation**: ±5% of frame size
- **Noise**: Gaussian noise (σ = 0.01)

#### 6.2 Temporal Augmentations
- **Temporal Jitter**: Random frame dropping/duplication
- **Speed Variation**: 0.8x to 1.2x playback speed

#### 6.3 Mirroring
- **Horizontal Flip**: Mirror landmarks and swap left/right labels
- **Purpose**: Create symmetric sign variants

### Configuration
```python
from preprocessing.augmentation import LandmarkAugmenter

augmenter = LandmarkAugmenter(
    rotation_range=15.0,
    scale_range=(0.9, 1.1),
    translation_range=0.05,
    noise_std=0.01,
    augmentation_probability=0.5
)

augmented_sequence = augmenter.augment_sequence(sequence)
```

---

## Step 7: Serialization

### Purpose
Save processed data in efficient format for training and inference.

### Storage Format

#### 7.1 NumPy Arrays (.npy)
- **Advantages**: Fast loading, compact
- **Structure**: (num_frames, num_landmarks * 3)
- **Use Case**: Processed landmark sequences

#### 7.2 JSON Metadata
- **Content**: Sign labels, frame indices, quality metrics
- **Use Case**: Dataset organization and debugging

### File Organization
```
data/processed/
├── train/
│   ├── A_001.npy
│   ├── A_002.npy
│   └── metadata.json
├── val/
│   └── ...
└── test/
    └── ...
```

---

## Data Statistics

### Expected Output
After preprocessing, the dataset should have:

- **Total Sequences**: Varies by dataset
- **Sequence Length**: 10-100 frames (variable)
- **Feature Dimension**: 63 (21 landmarks × 3 coords) per hand
- **Data Splits**:
  - Training: 70%
  - Validation: 15%
  - Test: 15%

### Quality Metrics
- **Valid Sequences**: \> 95% of input videos
- **Average Confidence**: \> 0.8 for landmark detection
- **Missing Data**: \< 5% per sequence

---

## Troubleshooting

### Common Issues

#### Issue 1: Low Landmark Detection Rate
- **Cause**: Poor lighting, occlusions, low resolution
- **Solution**: Adjust quality thresholds, improve video quality

#### Issue 2: Excessive Segmentation
- **Cause**: Motion threshold too low
- **Solution**: Increase `motion_threshold` parameter

#### Issue 3: Missing Segments
- **Cause**: Motion threshold too high
- **Solution**: Decrease `motion_threshold` parameter

#### Issue 4: Noisy Landmarks
- **Cause**: Insufficient filtering
- **Solution**: Increase filter window size, apply stronger smoothing

---

## Performance Benchmarks

### Processing Speed
- **Frame Extraction**: ~100 fps
- **Landmark Detection**: ~30 fps (CPU), ~60 fps (GPU)
- **Normalization**: ~1000 fps
- **Segmentation**: ~500 fps

### Resource Requirements
- **RAM**: ~4 GB for processing
- **Storage**: ~10 MB per minute of video (processed)
- **GPU**: Optional but recommended for faster landmark detection

---

## References

1. MediaPipe Hand Landmark Detection: https://google.github.io/mediapipe/solutions/hands.html
2. MediaPipe Pose Estimation: https://google.github.io/mediapipe/solutions/pose.html
3. Data Augmentation for Time Series: https://arxiv.org/abs/2002.12478

---

**Last Updated**: January 2026  
**Version**: 1.0
