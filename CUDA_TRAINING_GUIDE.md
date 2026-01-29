# Sign2Sound - CUDA-Accelerated Training Setup

**For PC with 8GB VRAM + CUDA - Optimized for GPU Training**

## 🚀 Quick Copy-Paste Prompt for Antigravity

```
Set up and train Sign2Sound ASL model with CUDA acceleration (8GB VRAM):

STEP 1: Verify CUDA
nvidia-smi
nvcc --version

STEP 2: Clone repo
git clone https://github.com/Atul013/SIGN2SOUND_ABH.git
cd SIGN2SOUND_ABH
git checkout Dev-B

STEP 3: Setup with PyTorch CUDA
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

STEP 4: Download MediaPipe
wget -O models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

STEP 5: Download dataset
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d ASL/

STEP 6: Preprocess (~45 min)
python scripts/preprocess_asl_images.py --train-only --create-splits
python scripts/preprocess_asl_images.py --test-only

STEP 7: Train with CUDA (~2-3 hours, batch_size=32 for 8GB)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python training/train.py --config training/config.yaml --use-cuda

Monitor GPU: watch -n 1 nvidia-smi

STEP 8: Evaluate
python training/evaluate.py --model checkpoints/best_model.h5 --use-cuda

STEP 9: Test real-time
python inference/realtime_demo.py --model checkpoints/best_model.h5 --use-cuda

Expected: 90-95% accuracy, <15ms inference, ~6-7GB GPU usage
```

---

## 📊 Performance with 8GB VRAM + CUDA

| Metric | CPU Only | 8GB CUDA |
|--------|----------|----------|
| Training Time | 8-10 hours | 2-3 hours |
| Inference Speed | 50ms | 15ms |
| Batch Size | 16 | 32 |
| Speedup | 1x | **3-4x faster** |

---

## 🔧 CUDA Optimization Tips

### Memory Management (8GB VRAM)
```bash
# Keep batch_size at 32 (safe for 8GB)
# Enable gradient checkpointing
# Use mixed precision for 2x speedup
export TF_ENABLE_AUTO_MIXED_PRECISION=1
```

### Monitor GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Check memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Troubleshooting

**CUDA not detected:**
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory:**
```bash
# Reduce batch_size to 16
# Clear cache: python -c "import torch; torch.cuda.empty_cache()"
```

**CUDA version mismatch:**
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## ✅ Expected Results

- ✅ Training: 2-3 hours (vs 8+ on CPU)
- ✅ Accuracy: 90-95%
- ✅ Inference: <15ms per frame
- ✅ GPU Usage: 6-7GB / 8GB

**Save and use this on your CUDA-enabled PC!**
