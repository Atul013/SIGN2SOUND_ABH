# Sign2Sound - Complete Training Setup for High-Performance PC

**For PC with 8GB VRAM - Full Pipeline from Clone to Trained Model**

Copy and paste this entire prompt into Antigravity on your high-performance PC.

---

## 🎯 PROMPT FOR ANTIGRAVITY

```
I need you to set up and train the Sign2Sound ASL Recognition model on this PC. This PC has 8GB VRAM and should provide faster training. Please follow these steps:

## Step 1: Clone Repository

Clone the Sign2Sound repository and checkout the Dev-B branch:

```bash
git clone https://github.com/Atul013/SIGN2SOUND_ABH.git
cd SIGN2SOUND_ABH
git checkout Dev-B
```

## Step 2: Environment Setup

Create a virtual environment and install all dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Install GPU-accelerated TensorFlow (for 12GB VRAM)
pip install tensorflow-gpu==2.13.0

# Verify GPU is detected
python -c "import tensorflow as tf; print('GPUs Available:', tf.config.list_physical_devices('GPU'))"
```

## Step 3: Download Required Models

Download the MediaPipe hand landmarker model:

```bash
# Create models directory
mkdir -p models

# Download MediaPipe model (Linux/Mac)
wget -O models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Or using curl
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models/hand_landmarker.task"
```

## Step 4: Download ASL Dataset

Download the ASL Alphabet Dataset from Kaggle:

**Option A: Using Kaggle API (Recommended)**
```bash
# Install Kaggle API
pip install kaggle

# Setup Kaggle credentials (place kaggle.json in ~/.kaggle/)
# Download from: https://www.kaggle.com/settings -> Create New API Token

# Download dataset
kaggle datasets download -d grassknoted/asl-alphabet

# Extract
unzip asl-alphabet.zip -d ASL/
```

**Option B: Manual Download**
1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Download the dataset
3. Extract to: `ASL/ASL_Alphabet_Dataset/`

Expected structure:
```
ASL/
└── ASL_Alphabet_Dataset/
    ├── asl_alphabet_train/
    │   ├── A/
    │   ├── B/
    │   └── ... (29 classes total)
    └── asl_alphabet_test/
        ├── A/
        ├── B/
        └── ... (29 classes total)
```

## Step 5: Verify Dataset Structure

Check that the dataset is properly structured:

```bash
# List training classes
ls ASL/ASL_Alphabet_Dataset/asl_alphabet_train/

# Should show: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z del nothing space

# Count images in one class (should be ~3000)
ls ASL/ASL_Alphabet_Dataset/asl_alphabet_train/A/ | wc -l
```

## Step 6: Preprocess Dataset

Run the preprocessing script to extract hand landmarks from all images:

```bash
# Preprocess training images and create train/val splits
python scripts/preprocess_asl_images.py --train-only --create-splits

# This will:
# - Extract hand landmarks from ~87,000 training images
# - Normalize landmarks (wrist-relative)
# - Save as .npy files in data/processed/
# - Create 85/15 train/validation split
# - Take approximately 45-60 minutes

# After training preprocessing completes, preprocess test set
python scripts/preprocess_asl_images.py --test-only

# This will process ~870 test images (~2-3 minutes)
```

**Expected output structure:**
```
data/processed/
├── train/          # 85% of training data
│   ├── A/
│   ├── B/
│   └── ... (29 classes, .npy files)
├── val/            # 15% of training data
│   ├── A/
│   ├── B/
│   └── ... (29 classes, .npy files)
└── test/           # Test set
    ├── A/
    ├── B/
    └── ... (29 classes, .npy files)
```

## Step 7: Verify Preprocessing

Check preprocessing results:

```bash
# Check preprocessing statistics
cat data/processed/train/preprocessing_stats.json

# Count processed files
find data/processed/train -name "*.npy" | wc -l  # Should be ~74,000
find data/processed/val -name "*.npy" | wc -l    # Should be ~13,000
find data/processed/test -name "*.npy" | wc -l   # Should be ~800
```

## Step 8: Configure Training for GPU

The training configuration is already optimized, but verify GPU settings:

```bash
# Check training/config.yaml
cat training/config.yaml

# Key settings for 12GB VRAM:
# - batch_size: 32 (can increase to 64 or 128 with 12GB VRAM)
# - epochs: 100
# - model: GRU with 128 hidden units
```

**Optional: Increase batch size for faster training**

Edit `training/config.yaml` and change:
```yaml
training:
  batch_size: 64  # Increase from 32 to 64 or 128
```

## Step 9: Train the Model

Start training with GPU acceleration:

```bash
# Train the model
python training/train.py --config training/config.yaml

# Training will:
# - Use GPU automatically if available
# - Train for up to 100 epochs (with early stopping)
# - Save best model to checkpoints/best_model.h5
# - Save final model to checkpoints/final_model.h5
# - Log metrics to results/training_log.txt
# - Generate plots in results/
# - Take approximately 2-4 hours with GPU

# Monitor training progress in real-time
tail -f results/training_log.txt
```

**Expected training output:**
```
Epoch 1/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2300/2300 [==============================] - 45s 20ms/step - loss: 0.8234 - accuracy: 0.7456 - val_loss: 0.4123 - val_accuracy: 0.8734
Epoch 2/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2300/2300 [==============================] - 42s 18ms/step - loss: 0.3456 - accuracy: 0.8923 - val_loss: 0.2345 - val_accuracy: 0.9234
...
```

## Step 10: Evaluate the Model

After training completes, evaluate on test set:

```bash
# Evaluate model
python training/evaluate.py --model checkpoints/best_model.h5

# This will:
# - Load the best model
# - Evaluate on test set
# - Generate confusion matrix (results/confusion_matrix.png)
# - Calculate per-class metrics (results/per_class_performance.csv)
# - Create accuracy/loss curves (results/accuracy_curves.png, results/loss_curves.png)
# - Save sample predictions (results/sample_outputs/)
```

## Step 11: Test Real-time Inference

Test the trained model with real-time webcam:

```bash
# Run real-time demo
python inference/realtime_demo.py --model checkpoints/best_model.h5

# This will:
# - Open webcam
# - Detect hand landmarks in real-time
# - Predict ASL letters
# - Display predictions with confidence
# - Convert to speech (TTS)
```

## Step 12: Run the Web UI

Launch the web interface to showcase the model:

```bash
cd ui
python app.py

# Open browser: http://localhost:5000
# The UI will show:
# - Live demo with webcam
# - Training metrics
# - Real-time predictions
# - Text-to-speech
```

## Step 13: Run Tests (Optional)

Verify everything is working:

```bash
# Run all tests
python tests/test_model.py
python tests/test_inference.py
python tests/test_preprocessing.py

# Or run all at once
python -m pytest tests/ -v
```

## Expected Results

After completing all steps, you should have:

✅ **Preprocessing Complete**
- ~87,000 training images processed
- ~870 test images processed
- Hand landmarks extracted and normalized
- Train/val/test splits created

✅ **Model Trained**
- Best model saved to `checkpoints/best_model.h5`
- Training accuracy: 95-98%
- Validation accuracy: 90-95%
- Test accuracy: 88-93%

✅ **Evaluation Complete**
- Confusion matrix generated
- Per-class performance metrics
- Accuracy/loss curves
- Sample predictions

✅ **Real-time Demo Working**
- Webcam inference functional
- Sub-50ms prediction time
- Text-to-speech working

## Performance Expectations with 12GB VRAM

With your GPU, expect:
- **Preprocessing**: 30-45 minutes (CPU-bound, MediaPipe)
- **Training**: 1.5-3 hours (GPU-accelerated)
- **Evaluation**: 2-5 minutes
- **Inference**: <20ms per frame (real-time)

## Troubleshooting

**GPU not detected:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall TensorFlow GPU
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow-gpu==2.13.0
```

**Out of memory:**
```bash
# Reduce batch size in training/config.yaml
batch_size: 32  # or even 16
```

**Preprocessing too slow:**
```bash
# Process in smaller batches
python scripts/preprocess_asl_images.py --max-samples 1000 --train-only
```

## Summary

This will give you a fully trained ASL recognition model with:
- 29 classes (A-Z + del, nothing, space)
- 90%+ accuracy
- Real-time inference capability
- Beautiful web UI
- Complete documentation

Please execute these steps in order and let me know if you encounter any issues!
```

---

## 📋 QUICK COPY-PASTE VERSION

If you want a shorter version, use this:

```
Set up and train Sign2Sound ASL model on this PC (12GB VRAM):

1. Clone repo:
git clone https://github.com/Atul013/SIGN2SOUND_ABH.git
cd SIGN2SOUND_ABH
git checkout Dev-B

2. Setup environment:
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install tensorflow-gpu==2.13.0

3. Download MediaPipe model:
wget -O models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

4. Download ASL dataset from Kaggle:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
Extract to: ASL/ASL_Alphabet_Dataset/

5. Preprocess data:
python scripts/preprocess_asl_images.py --train-only --create-splits
python scripts/preprocess_asl_images.py --test-only

6. Train model (with GPU):
python training/train.py --config training/config.yaml

7. Evaluate:
python training/evaluate.py --model checkpoints/best_model.h5

8. Test real-time:
python inference/realtime_demo.py --model checkpoints/best_model.h5

9. Run UI:
cd ui && python app.py

Expected time: ~2-3 hours total (preprocessing + training)
Expected accuracy: 90-95%
```

---

## 🎯 ULTRA-SHORT VERSION (One Command)

For maximum automation, use this single command:

```bash
git clone https://github.com/Atul013/SIGN2SOUND_ABH.git && cd SIGN2SOUND_ABH && git checkout Dev-B && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && pip install tensorflow-gpu && python scripts/preprocess_asl_images.py && python training/train.py
```

(Note: You'll still need to download the dataset and MediaPipe model manually)

---

## 📊 Expected Timeline

| Step | Time | GPU Usage |
|------|------|-----------|
| Clone & Setup | 5 min | - |
| Download Dataset | 10 min | - |
| Download MediaPipe | 1 min | - |
| Preprocessing | 45 min | Low (MediaPipe uses CPU) |
| Training | 2 hours | High (12GB VRAM fully utilized) |
| Evaluation | 3 min | Medium |
| Testing | 5 min | Low |
| **Total** | **~3 hours** | - |

---

## 💡 Tips for 12GB VRAM

1. **Increase batch size** to 64 or 128 for faster training
2. **Enable mixed precision** for even faster training:
   ```python
   # Add to training/train.py
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```
3. **Use larger model** if needed (more layers, more units)
4. **Enable TensorBoard** for monitoring:
   ```bash
   tensorboard --logdir=results/tensorboard
   ```

---

**Save this file and use it on your high-performance PC!**
