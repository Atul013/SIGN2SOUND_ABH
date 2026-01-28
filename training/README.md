# Training Module - ASL Alphabet Recognition

This directory contains all training-related code for the ASL alphabet recognition model.

## Files

- **`train.py`** - Main training script
- **`data_loader.py`** - Data loading and batching utilities
- **`config.yaml`** - Training configuration file
- **`callbacks.py`** - Training callbacks (early stopping, LR scheduling, etc.)
- **`evaluate.py`** - Model evaluation script
- **`README.md`** - This file

## Quick Start

### 1. Prepare Dataset

First, download and preprocess the ASL alphabet dataset:

```bash
# Download dataset
bash scripts/download_datasets.sh

# Preprocess with Developer A's pipeline
python preprocessing/preprocess.py --input data/raw --output data/processed
```

### 2. Configure Training

Edit `training/config.yaml` to set your training parameters:

```yaml
model:
  type: "gru"  # Options: "cnn", "gru", "lstm"
  
data:
  batch_size: 32
  max_sequence_length: 30

training:
  epochs: 100
  learning_rate: 0.001
```

### 3. Train Model

```bash
python training/train.py
```

### 4. Evaluate Model

```bash
python training/evaluate.py
```

## Training Configuration

### Model Types

1. **CNN** - For static letters (A-Z except J)
   - Fastest training (~2 hours)
   - Smallest model (~2-3 MB)
   - Best accuracy for static poses (>95%)

2. **GRU** - For all letters including J
   - Medium training time (~4 hours)
   - Medium model size (~4-5 MB)
   - Good for temporal sequences (>90%)

3. **LSTM** - Hybrid approach
   - Longest training (~5 hours)
   - Largest model (~6-8 MB)
   - Best overall performance (>92%)

### Hyperparameters

Key hyperparameters to tune:

- **Learning Rate**: Start with 0.001, reduce if not converging
- **Batch Size**: 32 is good default, increase if you have more GPU memory
- **Hidden Size**: 128 for GRU/LSTM, can increase to 256 for better accuracy
- **Dropout**: 0.3 is good default, increase to 0.5 if overfitting

### Data Augmentation

Enabled by default for training:
- Random scaling (0.9-1.1x)
- Random noise (std=0.01)
- Temporal shift (±2 frames)

Disable in `config.yaml` if needed:
```yaml
data:
  augmentation:
    enabled: false
```

## Training Callbacks

### Early Stopping

Stops training when validation accuracy stops improving:

```yaml
training:
  early_stopping:
    enabled: true
    patience: 10  # Stop after 10 epochs without improvement
    min_delta: 0.001  # Minimum improvement threshold
```

### Learning Rate Scheduling

Reduces learning rate when validation loss plateaus:

```yaml
training:
  lr_schedule:
    type: "reduce_on_plateau"
    factor: 0.5  # Reduce LR by 50%
    patience: 5  # After 5 epochs without improvement
```

### Model Checkpointing

Saves best model during training:

```yaml
training:
  checkpoint:
    save_best_only: true
    monitor: "val_accuracy"
    mode: "max"
```

## Evaluation Metrics

The evaluation script computes:

1. **Overall Accuracy** - Percentage of correct predictions
2. **Precision** - True positives / (True positives + False positives)
3. **Recall** - True positives / (True positives + False negatives)
4. **F1 Score** - Harmonic mean of precision and recall
5. **Per-Class Metrics** - Accuracy for each letter (A-Z)
6. **Confusion Matrix** - Shows which letters are confused

Results are saved to:
- `results/metrics.json` - All metrics in JSON format
- `results/per_class_metrics.csv` - Per-letter accuracy
- `results/confusion_matrix.npy` - Confusion matrix

## Monitoring Training

### TensorBoard (if enabled)

```bash
tensorboard --logdir results/logs
```

### Training Logs

Check `results/training_log.txt` for detailed logs.

### Training History

View `results/training_history.json` for epoch-by-epoch metrics.

## Troubleshooting

### Low Accuracy

1. **Check data quality**
   - Verify landmarks are extracted correctly
   - Check for missing/corrupted samples

2. **Increase model capacity**
   - Increase hidden_size (128 → 256)
   - Add more layers (2 → 3)

3. **Adjust learning rate**
   - Try lower LR (0.001 → 0.0001)
   - Enable LR scheduling

4. **Train longer**
   - Increase epochs (100 → 200)
   - Disable early stopping temporarily

### Overfitting

1. **Increase regularization**
   - Increase dropout (0.3 → 0.5)
   - Add L2 regularization

2. **More data augmentation**
   - Increase augmentation strength
   - Add more augmentation types

3. **Reduce model capacity**
   - Decrease hidden_size (256 → 128)
   - Remove layers

### Slow Training

1. **Reduce batch size** if running out of memory
2. **Use CNN model** instead of LSTM/GRU
3. **Enable mixed precision** training (if supported)
4. **Reduce max_sequence_length** (30 → 20)

## Best Practices

1. **Start Simple**
   - Begin with CNN model on static letters
   - Achieve high accuracy before adding complexity

2. **Monitor Validation**
   - Always check validation metrics
   - Stop if validation accuracy plateaus

3. **Save Checkpoints**
   - Keep best model checkpoint
   - Save periodic checkpoints for recovery

4. **Log Everything**
   - Enable detailed logging
   - Track all hyperparameters

5. **Validate on Test Set**
   - Only evaluate on test set once
   - Use validation set for hyperparameter tuning

## Integration with Developer A

Training uses preprocessed data from Developer A's pipeline:

```python
# Developer A provides:
- Hand landmark extraction (features/hand_landmarks.py)
- Temporal segmentation (preprocessing/temporal_segmentation.py)
- Robustness filtering (preprocessing/robustness.py)

# Developer B uses:
- Preprocessed landmark sequences (data/processed/)
- Standardized data format (features/feature_utils.py)
```

## Next Steps

After training:

1. **Evaluate on test set**
   ```bash
   python training/evaluate.py
   ```

2. **Test inference**
   ```bash
   python inference/infer.py
   ```

3. **Run real-time demo**
   ```bash
   python inference/realtime_demo.py
   ```

4. **Deploy model**
   - Export to ONNX/TFLite for production
   - Optimize for inference speed
   - Package with application

## Support

For issues or questions:
1. Check configuration in `config.yaml`
2. Review logs in `results/`
3. Verify data in `data/processed/`
4. Test with sample data first
