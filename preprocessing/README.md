# Preprocessing Module

This module contains scripts for preprocessing video data and extracting temporal features for sign language recognition.

## Files

### `temporal_segmentation.py`
Segments continuous landmark sequences into individual signs based on motion analysis.

**Key Features:**
- Frame-to-frame velocity computation
- Motion boundary detection (sign start/end)
- Pause detection for phrase boundaries
- Configurable thresholds for motion and pause detection

**Usage:**
```python
from preprocessing.temporal_segmentation import TemporalSegmenter

segmenter = TemporalSegmenter(
    motion_threshold=0.02,
    min_sign_duration=10,
    pause_duration=5
)

# landmarks_sequence is a list of landmark frames
segments = segmenter.segment_sequence(landmarks_sequence)

for seg in segments:
    print(f"Sign: frames {seg['start_frame']}-{seg['end_frame']}")
```

### `sequence_buffer.py`
Manages sliding window buffers for real-time landmark processing.

**Key Features:**
- Fixed-size circular buffer for continuous input
- Overlapping window creation for batch processing
- Real-time frame processing
- Buffer overflow handling

**Usage:**
```python
from preprocessing.sequence_buffer import SequenceBuffer, RealTimeProcessor

# For batch processing
buffer = SequenceBuffer(max_buffer_size=100, overlap_frames=10)
buffer.add_frame(landmarks)
windows = buffer.create_overlapping_windows(window_size=30)

# For real-time processing
processor = RealTimeProcessor(buffer_size=100)
result = processor.process_frame(landmarks)
```

## Pipeline Overview

```
Video Input
    ↓
Feature Extraction (hand_landmarks.py)
    ↓
Sequence Buffering (sequence_buffer.py)
    ↓
Temporal Segmentation (temporal_segmentation.py)
    ↓
Segmented Sign Sequences
```

## Configuration

Default parameters are tuned for 30fps video input:
- `motion_threshold`: 0.02 (normalized coordinates)
- `min_sign_duration`: 10 frames (~0.33 seconds)
- `pause_duration`: 5 frames (~0.17 seconds)
- `buffer_size`: 100 frames (~3.3 seconds)

Adjust these based on your specific use case and video frame rate.
