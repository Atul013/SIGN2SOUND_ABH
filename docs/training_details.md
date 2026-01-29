# Training Details

## Overview

This document provides comprehensive details about the model training process for Sign2Sound Phase 2, including architecture choices, hyperparameters, training strategies, and optimization techniques.

---

## Model Architecture

### Base Architecture: LSTM/GRU

The primary model uses recurrent neural networks (RNN) for temporal sequence modeling:

```
Input: (batch_size, sequence_length, 63)
    ↓
LSTM/GRU Layer 1 (128 units, bidirectional)
    ↓
Dropout (0.3)
    ↓
LSTM/GRU Layer 2 (128 units)
    ↓
Dropout (0.3)
    ↓
Dense Layer (256 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (26 units, Softmax)
```

### Architecture Variants

#### 1. Lightweight LSTM (Edge Deployment)
- **Hidden Size**: 64
- **Layers**: 1
- **Parameters**: ~150K
- **Model Size**: ~0.6 MB
- **Inference Time**: \< 50ms (CPU)

#### 2. Standard GRU (Balanced)
- **Hidden Size**: 128
- **Layers**: 2
- **Parameters**: ~500K
- **Model Size**: ~2 MB
- **Inference Time**: \< 100ms (CPU)

#### 3. Deep LSTM (High Accuracy)
- **Hidden Size**: 256
- **Layers**: 3
- **Parameters**: ~2M
- **Model Size**: ~8 MB
- **Inference Time**: ~200ms (CPU)

---

## Training Configuration

### Hyperparameters

```yaml
# From training/config.yaml

model:
  type: gru
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: false

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  
  # Learning rate scheduling
  lr_scheduler:
    type: reduce_on_plateau
    factor: 0.5
    patience: 5
    min_lr: 0.00001
  
  # Early stopping
  early_stopping:
    patience: 10
    min_delta: 0.001
    monitor: val_accuracy

data:
  sequence_length: 30  # Fixed length padding/truncation
  augmentation: true
  augmentation_probability: 0.5
  
loss:
  type: categorical_crossentropy
  label_smoothing: 0.1
  
regularization:
  dropout: 0.3
  l2_weight: 0.0001
```

### Data Loading

#### Sequence Preprocessing
1. **Padding**: Sequences shorter than 30 frames are padded with edge values
2. **Truncation**: Sequences longer than 30 frames are truncated
3. **Normalization**: Features normalized to zero mean, unit variance

#### Batch Composition
- **Stratified Sampling**: Ensures balanced class distribution per batch
- **Shuffle**: Training data shuffled each epoch
- **Prefetch**: Data loaded asynchronously for GPU efficiency

---

## Training Procedure

### Phase 1: Initial Training (Epochs 1-20)

**Objective**: Learn basic sign patterns

- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Augmentation**: Enabled
- **Focus**: Maximize training accuracy

### Phase 2: Fine-Tuning (Epochs 21-60)

**Objective**: Improve generalization

- **Learning Rate**: Reduced to 0.0005 (via scheduler)
- **Batch Size**: 32
- **Augmentation**: Enabled
- **Focus**: Minimize validation loss

### Phase 3: Convergence (Epochs 61-100)

**Objective**: Final optimization

- **Learning Rate**: Reduced to 0.0001
- **Batch Size**: 32
- **Augmentation**: Reduced probability (0.3)
- **Focus**: Stabilize performance

---

## Optimization Strategies

### 1. Optimizer: Adam

**Configuration**:
```python
optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)
```

**Rationale**: Adam combines momentum and adaptive learning rates, well-suited for RNNs.

### 2. Learning Rate Scheduling

**Strategy**: ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5
)
```

**Behavior**:
- Reduces LR by 50% if validation loss doesn't improve for 5 epochs
- Prevents premature convergence
- Minimum LR: 0.00001

### 3. Gradient Clipping

**Configuration**:
```python
clip_norm = 1.0
```

**Purpose**: Prevents exploding gradients in RNNs

---

## Regularization Techniques

### 1. Dropout

- **Rate**: 0.3
- **Locations**: After each LSTM/GRU layer and dense layer
- **Purpose**: Prevent overfitting

### 2. Label Smoothing

- **Smoothing Factor**: 0.1
- **Formula**: `y_smooth = y * 0.9 + 0.1 / num_classes`
- **Purpose**: Prevent overconfident predictions

### 3. L2 Regularization

- **Weight**: 0.0001
- **Applied To**: Dense layers
- **Purpose**: Encourage smaller weights

### 4. Data Augmentation

- **Probability**: 0.5 during training
- **Techniques**: Rotation, scaling, translation, noise
- **Purpose**: Improve robustness to variations

---

## Loss Functions

### Primary Loss: Categorical Cross-Entropy

```python
loss = -Σ y_true * log(y_pred)
```

### With Label Smoothing

```python
y_smooth = y_true * (1 - ε) + ε / K
loss = -Σ y_smooth * log(y_pred)
```

Where:
- ε = 0.1 (smoothing factor)
- K = 26 (number of classes)

### Optional: Focal Loss (for Class Imbalance)

```python
focal_loss = -α * (1 - p_t)^γ * log(p_t)
```

Where:
- α = 0.25 (class balance factor)
- γ = 2.0 (focusing parameter)

---

## Training Callbacks

### 1. ModelCheckpoint

```python
checkpoint = ModelCheckpoint(
    filepath='checkpoints/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

**Saves**: Best model based on validation accuracy

### 2. EarlyStopping

```python
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)
```

**Stops**: Training if no improvement for 10 epochs

### 3. TensorBoard Logging

```python
tensorboard = TensorBoard(
    log_dir='logs/',
    histogram_freq=1,
    write_graph=True
)
```

**Logs**: Training metrics, loss curves, model graph

### 4. CSV Logger

```python
csv_logger = CSVLogger(
    filename='results/training_log.csv',
    append=True
)
```

**Saves**: Epoch-wise metrics to CSV

---

## Evaluation Metrics

### Training Metrics

1. **Training Loss**: Cross-entropy loss on training set
2. **Training Accuracy**: Percentage of correct predictions
3. **Learning Rate**: Current learning rate value

### Validation Metrics

1. **Validation Loss**: Loss on validation set
2. **Validation Accuracy**: Accuracy on validation set
3. **Per-Class Accuracy**: Accuracy for each sign class

### Test Metrics (Final Evaluation)

1. **Test Accuracy**: Overall accuracy
2. **Precision**: TP / (TP + FP) per class
3. **Recall**: TP / (TP + FN) per class
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Class-wise prediction distribution

---

## Training Best Practices

### 1. Data Quality

- **Minimum Samples per Class**: 50
- **Balanced Distribution**: Avoid severe class imbalance
- **Quality Control**: Remove corrupted or mislabeled samples

### 2. Hyperparameter Tuning

**Grid Search Parameters**:
- Hidden size: [64, 128, 256]
- Number of layers: [1, 2, 3]
- Dropout: [0.2, 0.3, 0.4]
- Learning rate: [0.0001, 0.001, 0.01]

**Validation Strategy**: 5-fold cross-validation

### 3. Monitoring

**Watch For**:
- **Overfitting**: Training accuracy \>\> validation accuracy
- **Underfitting**: Both accuracies low and not improving
- **Gradient Issues**: Loss becoming NaN or exploding

**Solutions**:
- Overfitting → Increase dropout, add regularization
- Underfitting → Increase model capacity, train longer
- Gradient issues → Reduce learning rate, apply gradient clipping

---

## Training Time Estimates

### Hardware Configurations

#### CPU (Intel i7)
- **Time per Epoch**: ~5 minutes
- **Total Training**: ~8 hours (100 epochs)

#### GPU (NVIDIA RTX 3060)
- **Time per Epoch**: ~30 seconds
- **Total Training**: ~50 minutes (100 epochs)

#### GPU (NVIDIA A100)
- **Time per Epoch**: ~10 seconds
- **Total Training**: ~17 minutes (100 epochs)

---

## Model Deployment Considerations

### Edge Deployment Constraints

1. **Model Size**: \< 10 MB
2. **Inference Time**: \< 100ms per prediction
3. **Memory**: \< 500 MB RAM

### Optimization Techniques

#### 1. Model Quantization
- **Method**: Post-training quantization (INT8)
- **Size Reduction**: ~75%
- **Accuracy Loss**: \< 2%

#### 2. Pruning
- **Method**: Magnitude-based weight pruning
- **Sparsity**: 50%
- **Speed Improvement**: ~30%

#### 3. Knowledge Distillation
- **Teacher**: Large model (256 hidden units)
- **Student**: Small model (64 hidden units)
- **Accuracy Retention**: ~95%

---

## Troubleshooting

### Issue 1: Training Loss Not Decreasing

**Possible Causes**:
- Learning rate too high/low
- Poor initialization
- Data quality issues

**Solutions**:
- Adjust learning rate (try 0.0001 or 0.01)
- Use different initialization (Xavier, He)
- Check data preprocessing

### Issue 2: Overfitting

**Symptoms**:
- Training accuracy \> 95%, validation accuracy \< 80%
- Large gap between training and validation loss

**Solutions**:
- Increase dropout to 0.4-0.5
- Add more data augmentation
- Reduce model capacity
- Add L2 regularization

### Issue 3: Poor Generalization

**Symptoms**:
- Good validation accuracy, poor test accuracy
- Model fails on real-world data

**Solutions**:
- Increase data diversity
- Apply stronger augmentation
- Collect more representative test data

---

## References

1. Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
2. Cho et al. (2014): "Learning Phrase Representations using RNN Encoder-Decoder"
3. Szegedy et al. (2016): "Rethinking the Inception Architecture" (Label Smoothing)
4. Lin et al. (2017): "Focal Loss for Dense Object Detection"

---

**Last Updated**: January 2026  
**Version**: 1.0
