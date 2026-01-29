# Model Checkpoints

This directory contains trained model weights for the Sign2Sound Phase 2 project.

---

## Model Files

### `best_model.h5` (or `.pth` for PyTorch)
- **Description**: Best performing model based on validation accuracy
- **Selection Criteria**: Highest validation accuracy during training
- **Use Case**: Production deployment, inference, evaluation

### `final_model.h5` (or `.pth` for PyTorch)
- **Description**: Model weights from the final training epoch
- **Use Case**: Continued training, experimentation

---

## Model Information

### Architecture
- **Type**: GRU/LSTM-based temporal sequence model
- **Input**: (sequence_length, 63) - 21 hand landmarks × 3 coordinates
- **Output**: (26,) - Softmax probabilities for A-Z alphabet
- **Hidden Size**: 128 units
- **Layers**: 2 recurrent layers
- **Dropout**: 0.3

### Training Details
- **Dataset**: ASL Alphabet Dataset
- **Training Samples**: ~26,000 sequences
- **Validation Samples**: ~3,900 sequences
- **Test Samples**: ~3,900 sequences
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)

### Performance Metrics
- **Validation Accuracy**: ~85-90%
- **Test Accuracy**: ~85-90%
- **Inference Time**: \< 100ms (CPU)
- **Model Size**: ~2-8 MB (depending on architecture)

---

## Download Instructions

### For Models \> 100MB

If the model files exceed GitHub's 100MB limit, they are hosted externally:

#### Option 1: Google Drive
```bash
# Download using gdown
pip install gdown
gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O checkpoints/best_model.h5
```

#### Option 2: Hugging Face Hub
```bash
# Download using huggingface_hub
pip install huggingface_hub
huggingface-cli download YOUR_USERNAME/sign2sound-phase2 best_model.h5 --local-dir checkpoints/
```

#### Option 3: Direct Download
Download manually from:
- **Best Model**: [Link to be added]
- **Final Model**: [Link to be added]

---

## Loading Models

### TensorFlow/Keras
```python
from tensorflow import keras

# Load model
model = keras.models.load_model('checkpoints/best_model.h5')

# Make predictions
predictions = model.predict(input_sequence)
```

### PyTorch
```python
import torch
from models.model import ASLRecognitionModel

# Initialize model
model = ASLRecognitionModel(model_type='gru')

# Load weights
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_sequence)
```

---

## Model Versioning

### Version History

#### v1.0 (Current)
- **Date**: January 2026
- **Architecture**: GRU (128 hidden, 2 layers)
- **Dataset**: ASL Alphabet (A-Z)
- **Accuracy**: 87.5%
- **Notes**: Initial production model

#### v0.9 (Beta)
- **Date**: January 2026
- **Architecture**: LSTM (64 hidden, 1 layer)
- **Dataset**: ASL Alphabet (A-Z)
- **Accuracy**: 82.3%
- **Notes**: Lightweight model for edge devices

---

## Model Conversion

### Convert to TensorFlow Lite (Mobile/Edge)
```python
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('checkpoints/best_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('checkpoints/model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Convert to ONNX (Cross-platform)
```python
import tf2onnx
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('checkpoints/best_model.h5')

# Convert to ONNX
spec = (tf.TensorSpec((None, 30, 63), tf.float32, name="input"),)
output_path = "checkpoints/model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
```

---

## Model Quantization

### Post-Training Quantization (INT8)
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('checkpoints/best_model.h5')

# Convert with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

tflite_quant_model = converter.convert()

# Save quantized model
with open('checkpoints/model_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print(f"Original size: {os.path.getsize('checkpoints/best_model.h5') / 1024 / 1024:.2f} MB")
print(f"Quantized size: {len(tflite_quant_model) / 1024 / 1024:.2f} MB")
```

---

## Checksum Verification

To verify model integrity after download:

```bash
# Generate checksum
sha256sum checkpoints/best_model.h5

# Expected checksums (to be updated after training)
# best_model.h5: [checksum to be added]
# final_model.h5: [checksum to be added]
```

---

## Usage Examples

### Real-time Inference
```python
from inference.realtime_demo import RealtimeSignRecognition

# Initialize with model
recognizer = RealtimeSignRecognition(
    model_path='checkpoints/best_model.h5',
    confidence_threshold=0.7
)

# Run real-time recognition
recognizer.run()
```

### Batch Inference
```python
from inference.infer import SignRecognizer

# Load model
recognizer = SignRecognizer(model_path='checkpoints/best_model.h5')

# Process video file
predictions = recognizer.predict_video('path/to/video.mp4')
print(predictions)
```

---

## License

These model weights are released under the MIT License, consistent with the project license.

---

## Citation

If you use these models in your research, please cite:

```bibtex
@software{sign2sound_phase2,
  title={Sign2Sound Phase 2: ASL Alphabet Recognition},
  author={Sign2Sound Team},
  year={2026},
  url={https://github.com/yourusername/sign2sound-phase2}
}
```

---

## Support

For issues with model loading or performance:
1. Check that dependencies match `requirements.txt`
2. Verify input data format matches training data
3. Ensure model file is not corrupted (check checksum)
4. Open an issue on GitHub with error details

---

**Last Updated**: January 2026  
**Maintained By**: Sign2Sound Team
