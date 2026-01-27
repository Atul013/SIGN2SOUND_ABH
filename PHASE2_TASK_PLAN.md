# Sign2Sound Phase 2 – Task Plan

**Team Size**: 2 Developers  
**Strategy**: Vertical Ownership (Temporal Intelligence vs Semantic Intelligence)  
**Branch Strategy**: Feature branches → Pull Requests → Main  
**Timeline**: Phase 2 Development Cycle

---

## 🎯 Team Split Overview

| Developer | Focus Area | Branch | Core Responsibility |
|-----------|-----------|--------|---------------------|
| **Developer A** | Temporal Intelligence & Robustness | `feature/temporal-intelligence` | Understanding time, motion, and sign boundaries |
| **Developer B** | Semantic Intelligence & Communication | `feature/semantic-intelligence` | Understanding meaning and communicating output |

---

## 👤 Developer A – Temporal Intelligence & Robustness

**Branch**: `feature/temporal-intelligence`  
**Mission**: Build the foundation for understanding continuous sign sequences with temporal awareness.

### Tasks

#### 1. Feature Extraction Setup
- [ ] **feat: Set up MediaPipe integration**
  - Install and configure MediaPipe Holistic
  - Test hand landmark detection on sample video
  - Verify 21-point hand landmark output
  - Document MediaPipe configuration in `features/README.md`

- [ ] **feat: Implement hand landmark extraction**
  - Create `features/hand_landmarks.py`
  - Implement `extract_hand_landmarks(video_path)` function
  - Add normalization (wrist-relative coordinates)
  - Handle missing/occluded hands gracefully
  - Add unit tests in `tests/test_features.py`

- [ ] **feat: Implement pose estimation**
  - Create `features/pose_estimation.py`
  - Extract upper body pose (shoulders, elbows, wrists)
  - Normalize pose landmarks relative to shoulder center
  - Document pose keypoint indices

- [ ] **feat: Define landmark data contract**
  - Create `features/feature_utils.py`
  - Define standard data format (JSON/NumPy schema)
  - Implement serialization/deserialization functions
  - Document data contract in `features/README.md`

#### 2. Temporal Segmentation
- [ ] **feat: Implement motion-based segmentation**
  - Create `preprocessing/temporal_segmentation.py`
  - Compute frame-to-frame velocity (landmark deltas)
  - Detect motion peaks and valleys
  - Identify sign start/end boundaries
  - Add configurable thresholds (motion_threshold, min_sign_duration)

- [ ] **feat: Add pause detection logic**
  - Implement silence-aware pause detection
  - Define pause criteria (low velocity + duration)
  - Mark phrase boundaries at pauses
  - Handle short pauses vs. phrase-ending pauses

- [ ] **feat: Implement sequence buffering**
  - Create sliding window buffer for continuous input
  - Handle overlapping sign sequences
  - Implement real-time buffering for webcam input
  - Add buffer overflow handling

#### 3. Robustness & Edge Cases
- [ ] **feat: Add noise filtering**
  - Implement temporal smoothing (moving average)
  - Filter jitter in landmark positions
  - Handle sudden camera movements

- [ ] **feat: Handle occlusions and missing data**
  - Detect missing landmarks (confidence scores)
  - Implement interpolation for short gaps
  - Reject sequences with excessive missing data

- [ ] **docs: Document temporal pipeline**
  - Create `docs/temporal_pipeline.md`
  - Explain segmentation algorithm
  - Document assumptions (frame rate, motion thresholds)
  - Add visual diagrams of temporal flow

#### 4. Integration & Testing
- [ ] **feat: Create preprocessing pipeline**
  - Create `preprocessing/preprocess.py`
  - Integrate landmark extraction + segmentation
  - Process video → segmented landmark sequences
  - Save processed data to `data/processed/`

- [ ] **feat: Add inference utilities**
  - Create `inference/utils.py`
  - Implement real-time landmark extraction
  - Add webcam frame capture utilities
  - Handle frame rate normalization

- [ ] **test: Add comprehensive tests**
  - Test landmark extraction on edge cases
  - Test segmentation on continuous sequences
  - Validate data contract compliance
  - Add integration tests for full pipeline

### Deliverables
✅ Segmented sign sequences from continuous video input  
✅ Pause-aware temporal logic for phrase boundaries  
✅ Robust landmark extraction with noise handling  
✅ Clear documentation on temporal modeling assumptions  

### Affected Folders
```
features/
preprocessing/
inference/utils.py
tests/test_features.py
tests/test_preprocessing.py
docs/temporal_pipeline.md
```

---

## 👤 Developer B – Semantic Intelligence & Communication

**Branch**: `feature/semantic-intelligence`  
**Mission**: Build the model, training pipeline, and output communication system.

### Tasks

#### 1. Vocabulary & Dataset Preparation
- [ ] **feat: Define Phase-2 vocabulary**
  - Select 10–15 functional signs (greetings, needs, yes/no)
  - Document vocabulary in `data/README.md`
  - List sign names, categories, and expected usage
  - Create label mapping (sign_name → class_id)

- [ ] **feat: Prepare dataset splits**
  - Download WLASL/MS-ASL subsets for selected signs
  - Create train/val/test splits (70/15/15)
  - Document dataset statistics in `data/statistics.txt`
  - Add dataset download script in `scripts/download_datasets.sh`

- [ ] **feat: Create data loader**
  - Create `training/data_loader.py`
  - Load preprocessed landmark sequences
  - Implement batching and shuffling
  - Add data augmentation hooks

#### 2. Model Architecture
- [ ] **feat: Design temporal model**
  - Create `models/model.py`
  - Implement lightweight LSTM/GRU architecture
  - Input: sequence of landmark frames (T × N features)
  - Output: softmax over vocabulary classes
  - Keep model size < 10MB

- [ ] **feat: Add custom layers (if needed)**
  - Create `models/custom_layers.py`
  - Implement attention mechanism (optional)
  - Add temporal pooling layers
  - Document architecture in `models/README.md`

- [ ] **docs: Create architecture diagram**
  - Create `docs/architecture_diagram.png`
  - Visualize model layers and data flow
  - Show input/output dimensions

#### 3. Training Pipeline
- [ ] **feat: Implement training script**
  - Create `training/train.py`
  - Load data, initialize model, define optimizer
  - Implement training loop with validation
  - Save best model checkpoint to `checkpoints/`

- [ ] **feat: Create hyperparameter config**
  - Create `training/config.yaml`
  - Define learning rate, batch size, epochs
  - Add model hyperparameters (hidden size, layers)
  - Document config options in `training/README.md`

- [ ] **feat: Add training callbacks**
  - Create `training/callbacks.py`
  - Implement early stopping
  - Add learning rate scheduling
  - Log metrics to `results/training_log.txt`

- [ ] **feat: Implement evaluation script**
  - Create `training/evaluate.py`
  - Compute accuracy, precision, recall, F1
  - Generate confusion matrix
  - Save metrics to `results/metrics.json`

#### 4. Inference & Output
- [ ] **feat: Implement single inference**
  - Create `inference/infer.py`
  - Load trained model from checkpoint
  - Run inference on segmented landmark sequence
  - Output predicted sign label

- [ ] **feat: Implement phrase construction**
  - Create phrase-level logic in `inference/infer.py`
  - Combine consecutive sign predictions
  - Handle repeated signs (temporal filtering)
  - Output complete sentence/phrase

- [ ] **feat: Integrate text-to-speech**
  - Create `inference/tts.py`
  - Integrate offline TTS engine (pyttsx3 / gTTS)
  - Convert predicted text to speech
  - Save audio output (optional)

- [ ] **feat: Create real-time demo**
  - Create `inference/realtime_demo.py`
  - Integrate with Developer A's temporal pipeline
  - Run live webcam → landmarks → model → TTS
  - Add visual feedback (display predictions on screen)

#### 5. Results & Visualization
- [ ] **feat: Generate performance metrics**
  - Run evaluation on test set
  - Save metrics to `results/metrics.json`
  - Create per-class performance CSV

- [ ] **feat: Create visualizations**
  - Generate `results/confusion_matrix.png`
  - Plot `results/loss_curves.png`
  - Plot `results/accuracy_curves.png`
  - Add sample predictions to `results/sample_outputs/`

- [ ] **docs: Create technical report**
  - Create `docs/technical_report.pdf`
  - Document model architecture, training details
  - Report performance metrics
  - Discuss limitations and future work

- [ ] **docs: Document system pipeline**
  - Create `docs/system_pipeline.png`
  - Show end-to-end flow (input → output)
  - Highlight integration points with Developer A's work

### Deliverables
✅ Trained sign recognition model (< 10MB)  
✅ Phrase-level text output from continuous signs  
✅ Spoken audio output via offline TTS  
✅ Comprehensive evaluation metrics and visualizations  

### Affected Folders
```
data/README.md
data/statistics.txt
models/
training/
inference/infer.py
inference/tts.py
inference/realtime_demo.py
results/
checkpoints/
docs/architecture_diagram.png
docs/system_pipeline.png
docs/technical_report.pdf
scripts/download_datasets.sh
```

---

## 🔄 Integration Points

Both developers must coordinate on these interfaces:

| Interface | Owner | Consumer | Contract |
|-----------|-------|----------|----------|
| **Landmark Data Format** | Developer A | Developer B | JSON/NumPy schema defined in `features/feature_utils.py` |
| **Segmented Sequences** | Developer A | Developer B | List of (start_frame, end_frame, landmarks) tuples |
| **Vocabulary Definition** | Developer B | Developer A | Sign names and class IDs in `data/README.md` |
| **Real-time Pipeline** | Both | Both | Developer A provides landmarks, Developer B runs inference |

### Coordination Checklist
- [ ] **Week 1**: Developer A defines landmark data contract → Developer B reviews
- [ ] **Week 2**: Developer B defines vocabulary → Developer A reviews
- [ ] **Week 3**: Developer A delivers segmented sequences → Developer B tests with model
- [ ] **Week 4**: Integration testing for real-time demo

---

## 📋 Shared Rules

### Git Workflow
1. **No direct commits to `main`**
2. **All work in feature branches**:
   - Developer A: `feature/temporal-intelligence`
   - Developer B: `feature/semantic-intelligence`
3. **Merge only via Pull Requests**
4. **Require 1 approval before merge**
5. **Avoid editing the same files** (coordinate on shared files)

### Commit Conventions
Use clear prefixes for all commits:
- `feat:` – New feature
- `fix:` – Bug fix
- `docs:` – Documentation only
- `test:` – Adding tests
- `refactor:` – Code refactoring

**Examples**:
```
feat: implement hand landmark extraction
fix: handle missing landmarks in segmentation
docs: add temporal pipeline documentation
test: add unit tests for feature extraction
```

### Code Review Guidelines
- Review PRs within 24 hours
- Check for code quality, documentation, and tests
- Verify no conflicts with own branch
- Test integration points before approving

---

## 🎯 Success Criteria

Phase 2 is complete when:

✅ **Developer A Deliverables**:
- [ ] Landmark extraction works on live webcam
- [ ] Temporal segmentation detects sign boundaries
- [ ] Pause detection identifies phrase breaks
- [ ] Temporal pipeline is documented

✅ **Developer B Deliverables**:
- [ ] Model achieves >80% accuracy on test set
- [ ] Inference runs in <100ms per sign
- [ ] TTS produces clear audio output
- [ ] All visualizations and metrics are generated

✅ **Integration**:
- [ ] Real-time demo works end-to-end (webcam → speech)
- [ ] Both branches merged to `main` via PRs
- [ ] All mandatory folders populated per skeleton
- [ ] Technical report and documentation complete

---

## 📅 Suggested Timeline

| Week | Developer A | Developer B |
|------|-------------|-------------|
| **Week 1** | Feature extraction + data contract | Vocabulary definition + dataset prep |
| **Week 2** | Temporal segmentation + pause detection | Model architecture + training pipeline |
| **Week 3** | Robustness + buffering | Inference + phrase construction |
| **Week 4** | Integration utilities + docs | TTS + real-time demo + results |
| **Week 5** | Integration testing + final docs | Integration testing + technical report |

---

## 🚀 Getting Started

### Developer A
```bash
# Create feature branch
git checkout -b feature/temporal-intelligence

# Start with feature extraction
# 1. Install MediaPipe
pip install mediapipe opencv-python

# 2. Create features/hand_landmarks.py
# 3. Test on sample video
# 4. Commit and push
git add features/
git commit -m "feat: implement hand landmark extraction"
git push origin feature/temporal-intelligence
```

### Developer B
```bash
# Create feature branch
git checkout -b feature/semantic-intelligence

# Start with vocabulary definition
# 1. Research WLASL/MS-ASL datasets
# 2. Select 10-15 functional signs
# 3. Document in data/README.md
# 4. Commit and push
git add data/README.md
git commit -m "feat: define Phase-2 vocabulary"
git push origin feature/semantic-intelligence
```

---

**Good luck, team! 🚀**  
Remember: Clear communication, clean commits, and coordinated integration points are key to success.
