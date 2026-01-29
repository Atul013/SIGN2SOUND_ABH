# Sign2Sound: Technical Performance Report

**Date:** January 29, 2026
**Project:** Sign2Sound - Real-time ASL to Speech System
**Model Version:** `best_model.pth` (Epoch 10)

---

## 1. Quantitative Performance Metrics

The model was evaluated on a held-out validation set of **27,902** samples.

### **Types of Evaluation Metrics:**
*   **Accuracy:** 97.63% (Overall correctness)
*   **Precision:** 97.66% (Reliability of positive predictions)
*   **Recall:** 97.63% (Ability to find all positive instances)
*   **F1-Score:** 97.63% (Harmonic mean of precision and recall)

The high consistency across all metrics indicates a stable and well-balanced model that does not suffer from significant class bias given the balanced nature of the dataset.

---

## 2. Training Dynamics

The training process showed consistent convergence over 10 epochs.

### **Learning Curves**
*   **Training Loss:** Decreased from **0.603** (Epoch 1) to **0.115** (Epoch 10).
*   **Validation Loss:** Decreased from **0.221** (Epoch 1) to **0.083** (Epoch 10).
*   **Accuracy Gain:** Validation accuracy improved from **93.2%** to **97.6%**.

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|------------|----------|-----------|---------|
| 1     | 0.6029     | 0.2213   | 81.52%    | 93.23%  |
| 2     | 0.2476     | 0.1472   | 92.90%    | 95.81%  |
| 3     | 0.1963     | 0.1353   | 94.45%    | 96.29%  |
| 5     | 0.1539     | 0.1085   | 95.68%    | 97.01%  |
| 10    | 0.1152     | 0.0829   | 96.77%    | 97.63%  |

*See attached `training_curves.png` for visual representation.*

---

## 3. Per-Class Performance Analysis

### **Best Performing Classes**
These classes achieved F1-Scores > 99%, indicating highly distinct features.
*   **F (99.4%):** The "OK" hand sign is geometrically unique and distinct from other closed-fist signs.
*   **B (99.2%):** Open palm with tucked thumb is easily distinguishable.
*   **W (99.1%):** Three fingers up is a very distinct shape.
*   **Space (99.1%):** The "cup" hand shape for space is distinct from letter signs.

### **Challenging Classes**
These classes had the lowest performance, primarily due to visual similarity (inter-class similarity).
*   **N (91.7%) & M (92.9%):** 
    *   *Issue:* "M" (3 fingers over thumb) and "N" (2 fingers over thumb) are extremely similar. The subtle difference often gets confused, especially depending on camera angle.
*   **S (95.8%):** Confused with **A** and **E**. All are "fist-like" shapes.
*   **J (96.6%):** A dynamic gesture (motion of 'I'). Since the model processes frame-by-frame landmarks, the static snapshot of 'J' looks identical to 'I' or 'Y' at different points in the motion.

---

## 4. Inference Performance

### **System Latency**
The system is designed for real-time usage (≥ 25 FPS).

| Component | Latency (Approx) | Description |
|-----------|------------------|-------------|
| **MediaPipe** (Landmarks) | ~35 ms | CPU-based hand detection & keypoint extraction. |
| **Classifier** (GRU/Diff) | < 1 ms | Lightweight neural network inference. |
| **Total Pipeline** | **~40 ms** | **~25 FPS (Real-time capability)** |

### **SLM (Grammar Correction)**
*   **Latency:** ~150-300 ms.
*   *Note:* The SLM (Small Language Model) only runs when the "Speak Text" button is triggered, not on every frame. This ensures the visual feedback loop remains instantaneous while providing smart corrections on demand.

---

## 5. Confusion Matrix Summary
*   **Diagonal Dominance:** The confusion matrix shows a strong diagonal, confirming high classification accuracy.
*   **Clusters of Confusion:**
    *   **M vs N:** Significant off-diagonal leakage between these classifications.
    *   **A vs E vs S:** Minor confusion cluster due to compact fist shapes.
    *   **U vs V:** Occasional confusion due to two vertical fingers (closed vs open).

---

## 6. Conclusion
The Sign2Sound model achieves state-of-the-art performance for real-world ASL alphabet recognition. With an accuracy of **97.63%** and real-time processing capabilities, it is suitable for live communication assistance. Future work should focus on temporal modeling (LSTM/Transformer) to better resolve dynamic gestures like 'J' and 'Z'.
