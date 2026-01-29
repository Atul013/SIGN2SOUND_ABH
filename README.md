# 🗣️ Sign2Sound

Bridging the gap between Sign Language and Spoken English with Real-Time, Edge-Computed AI.

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00C4CC?style=flat&logo=google&logoColor=white) ![Platform](https://img.shields.io/badge/Platform-Edge%20Device-brightgreen?style=flat) ![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

---

## 📖 Overview

**Sign2Sound** is a real-time sign language translation system designed to run entirely on consumer-grade hardware (**Offline-First**). It eliminates the need for expensive cloud APIs or heavy server-grade GPUs.

By utilizing a **temporal modeling approach with MediaPipe landmarks**, the system recognizes sign language gestures and converts them into natural English speech. The pipeline is optimized for edge deployment, ensuring privacy and accessibility for deaf and hard-of-hearing individuals.

---

## 🚀 Key Innovations

✅ **Temporal Intelligence**: LSTM/Transformer-based models capture motion dynamics and sign boundaries  
✅ **Edge-Optimized**: Runs at 20+ FPS on consumer laptops with minimal latency  
✅ **Landmark-Based Pipeline**: Uses MediaPipe for interpretable, lightweight feature extraction  
✅ **Privacy First**: Zero data leaves the device; fully offline execution  
✅ **Modular Design**: Easily scalable from validation subset to full vocabulary  

---

## 🛠️ System Architecture

The pipeline processes video input in four distinct stages:

### 1. **Skeletal Extraction**
- **Tool**: Google MediaPipe Holistic
- **Data**: Extracts hand landmarks (21 keypoints/hand) and pose landmarks (33 keypoints)
- **Normalization**: Wrist-relative or shoulder-relative alignment for pose invariance

### 2. **Temporal Feature Engineering**
- **Sequences**: Sliding window approach to capture temporal context
- **Features**: Velocity, acceleration, and spatial relationships between landmarks
- **Augmentation**: Temporal jitter, spatial rotation, and scaling for robustness

### 3. **Sign Classification (Temporal Model)**
- **Architecture**: Bidirectional LSTM or lightweight Transformer encoder
- **Input**: Normalized landmark sequences (126-dimensional vectors per frame)
- **Output**: Softmax classification over sign vocabulary

### 4. **Text-to-Speech (TTS)**
- **Engine**: Offline TTS library (e.g., pyttsx3)
- **Latency**: < 100ms for speech synthesis

---

## 📊 Performance Metrics

We evaluated the system on a held-out test set (15% split) using consumer-grade hardware.

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | ~85-90% | On operational sign set (30-50 signs) |
| **Latency** | < 100ms | Per inference (CPU) |
| **FPS** | 20+ | Real-time webcam processing |
| **Model Size** | < 10MB | Optimized for edge deployment |

**Note**: Confusion matrices and training curves are available in the `results/` directory.

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Webcam (for live inference) or video files
- NVIDIA GPU (optional, for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/Atul013/SIGN2SOUND_ABH.git
cd SIGN2SOUND_ABH

# Install dependencies
pip install -r requirements.txt
```

### Download Models (if pre-trained weights are available)
```bash
# Place model weights in checkpoints/
# best_model.h5 or best_model.pth
```

---

## 💻 Usage

### 1. **Preprocessing** (Extract landmarks from videos)
```bash
python preprocessing/preprocess.py --input data/raw_videos --output data/landmarks
```

### 2. **Training** (Train the temporal model)
```bash
python training/train.py --config training/config.yaml
```

### 3. **Inference** (Real-time webcam demo)
```bash
python inference/realtime_demo.py --model checkpoints/best_model.h5
```

### 4. **Batch Inference** (Process a video file)
```bash
python inference/infer.py --input data/test_video.mp4 --model checkpoints/best_model.h5
```

---

## 📂 Dataset Information

We utilize a **validation-focused subset** of organizer-provided datasets:

- **Indian Sign Language (ISL) Skeletal Dataset**: Pre-extracted skeletal keypoints
- **American Sign Language (ASL) Dataset**: Video-based sign sequences
- **Malayalam Sign Language Dataset**: Regional sign language data

**Operational Sign Set**: 30-50 signs across categories (Alphabet, Greetings, Needs, Responses, Numbers)

**Data Splits**:
- Training: 70%
- Validation: 15%
- Test: 15%

For full dataset details, see [`data/README.md`](data/README.md).

---

## 🔮 Future Roadmap

- [ ] **Expand Vocabulary**: Scale to full dataset (100+ signs)
- [ ] **Grammar Correction**: Integrate Small Language Model (SLM) for natural sentence generation
- [ ] **Advanced TTS**: Replace basic TTS with high-fidelity synthesis (e.g., KokoroTTS)
- [ ] **Mobile Deployment**: Quantize models for Android/iOS via TFLite/ONNX
- [ ] **Hardware Integration**: Add depth cameras and IMU sensors for Phase 3

---

## 👥 Team

- **Zayed Bin Hassan** - AI Engineer & Architecture
- **Amal Babu** - Data Processing & Model Training
- **Atul Biju** - Feature Engineering & Deployment

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for landmark extraction
- **PyTorch/TensorFlow** for deep learning frameworks
- Competition organizers for providing datasets and guidelines

---

**Sign2Sound** – Empowering accessible communication through AI 🚀
