"""
Custom loss functions for sign language recognition.
Includes specialized losses for temporal modeling and class imbalance.
"""

import numpy as np
from typing import Optional, Dict, List


class FocalLoss:
    """
    Focal Loss for addressing class imbalance.
    Down-weights easy examples and focuses on hard negatives.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        self.alpha = alpha
        self.gamma = gamma
        
    def get_config(self) -> Dict:
        """Get loss configuration."""
        return {
            'type': 'FocalLoss',
            'alpha': self.alpha,
            'gamma': self.gamma,
            'formula': 'FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)',
            'description': 'Focal loss for handling class imbalance by down-weighting easy examples'
        }
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute focal loss (pseudo-implementation).
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            Loss value
        """
        # Clip predictions to prevent log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute cross entropy
        ce = -y_true * np.log(y_pred)
        
        # Compute focal term: (1 - p_t)^gamma
        p_t = np.sum(y_true * y_pred, axis=-1)
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = self.alpha * focal_term * np.sum(ce, axis=-1)
        
        return np.mean(focal_loss)


class TemporalConsistencyLoss:
    """
    Encourages temporal consistency in predictions.
    Penalizes rapid changes in predictions between consecutive frames.
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize temporal consistency loss.
        
        Args:
            weight: Weight for temporal consistency term
        """
        self.weight = weight
        
    def get_config(self) -> Dict:
        """Get loss configuration."""
        return {
            'type': 'TemporalConsistencyLoss',
            'weight': self.weight,
            'formula': 'TCL = λ * Σ ||pred_t - pred_{t-1}||^2',
            'description': 'Penalizes inconsistent predictions across consecutive frames'
        }
    
    def compute(self, predictions_sequence: np.ndarray) -> float:
        """
        Compute temporal consistency loss.
        
        Args:
            predictions_sequence: Sequence of predictions (seq_len, num_classes)
            
        Returns:
            Loss value
        """
        if len(predictions_sequence) < 2:
            return 0.0
        
        # Compute differences between consecutive predictions
        diffs = predictions_sequence[1:] - predictions_sequence[:-1]
        
        # L2 norm of differences
        consistency_loss = np.mean(np.sum(diffs ** 2, axis=-1))
        
        return self.weight * consistency_loss


class LabelSmoothingLoss:
    """
    Label smoothing for better generalization.
    Prevents overconfident predictions.
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor (0 = no smoothing, 1 = uniform distribution)
        """
        self.smoothing = smoothing
        
    def get_config(self) -> Dict:
        """Get loss configuration."""
        return {
            'type': 'LabelSmoothingLoss',
            'smoothing': self.smoothing,
            'formula': 'y_smooth = y * (1 - ε) + ε / K',
            'description': 'Smooths one-hot labels to prevent overconfident predictions'
        }
    
    def smooth_labels(self, y_true: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Apply label smoothing to one-hot labels.
        
        Args:
            y_true: One-hot encoded labels (batch_size, num_classes)
            num_classes: Number of classes
            
        Returns:
            Smoothed labels
        """
        return y_true * (1 - self.smoothing) + self.smoothing / num_classes
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
        """
        Compute cross-entropy with label smoothing.
        
        Args:
            y_true: True labels (one-hot)
            y_pred: Predicted probabilities
            num_classes: Number of classes
            
        Returns:
            Loss value
        """
        # Smooth labels
        y_smooth = self.smooth_labels(y_true, num_classes)
        
        # Compute cross-entropy
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        ce = -np.sum(y_smooth * np.log(y_pred), axis=-1)
        
        return np.mean(ce)


class ContrastiveLoss:
    """
    Contrastive loss for metric learning.
    Useful for learning discriminative features between sign classes.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs
        """
        self.margin = margin
        
    def get_config(self) -> Dict:
        """Get loss configuration."""
        return {
            'type': 'ContrastiveLoss',
            'margin': self.margin,
            'formula': 'L = y * d^2 + (1-y) * max(margin - d, 0)^2',
            'description': 'Pulls similar pairs together, pushes dissimilar pairs apart'
        }
    
    def compute(self, embeddings1: np.ndarray, embeddings2: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute contrastive loss.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: 1 if same class, 0 if different
            
        Returns:
            Loss value
        """
        # Euclidean distance
        distances = np.sqrt(np.sum((embeddings1 - embeddings2) ** 2, axis=-1))
        
        # Contrastive loss
        positive_loss = labels * distances ** 2
        negative_loss = (1 - labels) * np.maximum(self.margin - distances, 0) ** 2
        
        return np.mean(positive_loss + negative_loss)


class TripletLoss:
    """
    Triplet loss for learning embeddings.
    Ensures anchor is closer to positive than to negative by a margin.
    """
    
    def __init__(self, margin: float = 0.5):
        """
        Initialize triplet loss.
        
        Args:
            margin: Minimum margin between positive and negative distances
        """
        self.margin = margin
        
    def get_config(self) -> Dict:
        """Get loss configuration."""
        return {
            'type': 'TripletLoss',
            'margin': self.margin,
            'formula': 'L = max(||a - p||^2 - ||a - n||^2 + margin, 0)',
            'description': 'Triplet loss: anchor closer to positive than negative'
        }
    
    def compute(self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive (same class) embeddings
            negative: Negative (different class) embeddings
            
        Returns:
            Loss value
        """
        # Distances
        pos_dist = np.sum((anchor - positive) ** 2, axis=-1)
        neg_dist = np.sum((anchor - negative) ** 2, axis=-1)
        
        # Triplet loss
        loss = np.maximum(pos_dist - neg_dist + self.margin, 0)
        
        return np.mean(loss)


class WeightedCrossEntropy:
    """
    Weighted cross-entropy for class imbalance.
    Assigns different weights to different classes.
    """
    
    def __init__(self, class_weights: Optional[Dict[int, float]] = None):
        """
        Initialize weighted cross-entropy.
        
        Args:
            class_weights: Dict mapping class indices to weights
        """
        self.class_weights = class_weights or {}
        
    def get_config(self) -> Dict:
        """Get loss configuration."""
        return {
            'type': 'WeightedCrossEntropy',
            'class_weights': self.class_weights,
            'formula': 'WCE = -Σ w_i * y_i * log(p_i)',
            'description': 'Cross-entropy with per-class weights for imbalance handling'
        }
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute weighted cross-entropy.
        
        Args:
            y_true: True labels (one-hot)
            y_pred: Predicted probabilities
            
        Returns:
            Loss value
        """
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Apply class weights
        weights = np.ones_like(y_true)
        for class_idx, weight in self.class_weights.items():
            weights[:, class_idx] = weight
        
        # Weighted cross-entropy
        ce = -weights * y_true * np.log(y_pred)
        
        return np.mean(np.sum(ce, axis=-1))


class CombinedLoss:
    """
    Combines multiple loss functions with configurable weights.
    """
    
    def __init__(self, losses: List[Dict]):
        """
        Initialize combined loss.
        
        Args:
            losses: List of dicts with 'loss' object and 'weight' float
        """
        self.losses = losses
        
    def get_config(self) -> Dict:
        """Get loss configuration."""
        return {
            'type': 'CombinedLoss',
            'num_losses': len(self.losses),
            'losses': [
                {
                    'loss_type': loss['loss'].get_config()['type'],
                    'weight': loss['weight']
                }
                for loss in self.losses
            ],
            'description': 'Weighted combination of multiple loss functions'
        }
    
    def compute(self, *args, **kwargs) -> float:
        """
        Compute combined loss.
        
        Returns:
            Weighted sum of all losses
        """
        total_loss = 0.0
        
        for loss_config in self.losses:
            loss_obj = loss_config['loss']
            weight = loss_config['weight']
            
            # Each loss computes its own value
            loss_value = loss_obj.compute(*args, **kwargs)
            total_loss += weight * loss_value
        
        return total_loss


def compute_class_weights(class_counts: Dict[int, int], method: str = 'balanced') -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        class_counts: Dict mapping class indices to sample counts
        method: Weighting method ('balanced', 'inverse', 'sqrt_inverse')
        
    Returns:
        Dict mapping class indices to weights
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = {}
    
    for class_idx, count in class_counts.items():
        if method == 'balanced':
            # Balanced: n_samples / (n_classes * n_samples_class)
            weights[class_idx] = total_samples / (num_classes * count)
        elif method == 'inverse':
            # Inverse frequency
            weights[class_idx] = total_samples / count
        elif method == 'sqrt_inverse':
            # Square root of inverse frequency (less aggressive)
            weights[class_idx] = np.sqrt(total_samples / count)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights to have mean = 1
    mean_weight = np.mean(list(weights.values()))
    weights = {k: v / mean_weight for k, v in weights.items()}
    
    return weights


def get_loss_registry() -> Dict[str, type]:
    """
    Get registry of all custom loss functions.
    
    Returns:
        Dict mapping loss names to loss classes
    """
    return {
        'FocalLoss': FocalLoss,
        'TemporalConsistencyLoss': TemporalConsistencyLoss,
        'LabelSmoothingLoss': LabelSmoothingLoss,
        'ContrastiveLoss': ContrastiveLoss,
        'TripletLoss': TripletLoss,
        'WeightedCrossEntropy': WeightedCrossEntropy,
        'CombinedLoss': CombinedLoss
    }


if __name__ == "__main__":
    print("Custom Loss Functions for Sign Language Recognition")
    print("=" * 70)
    
    # Demonstrate all loss functions
    registry = get_loss_registry()
    
    print(f"\nAvailable Loss Functions: {len(registry)}")
    print("-" * 70)
    
    for name, loss_class in registry.items():
        print(f"\n{name}:")
        
        # Create instance with default parameters
        if name == 'FocalLoss':
            loss = loss_class(alpha=0.25, gamma=2.0)
        elif name == 'TemporalConsistencyLoss':
            loss = loss_class(weight=0.1)
        elif name == 'LabelSmoothingLoss':
            loss = loss_class(smoothing=0.1)
        elif name == 'ContrastiveLoss':
            loss = loss_class(margin=1.0)
        elif name == 'TripletLoss':
            loss = loss_class(margin=0.5)
        elif name == 'WeightedCrossEntropy':
            loss = loss_class(class_weights={0: 1.0, 1: 2.0})
        elif name == 'CombinedLoss':
            loss = loss_class(losses=[
                {'loss': FocalLoss(), 'weight': 1.0},
                {'loss': TemporalConsistencyLoss(), 'weight': 0.1}
            ])
        
        config = loss.get_config()
        print(f"  Type: {config['type']}")
        print(f"  Description: {config['description']}")
        if 'formula' in config:
            print(f"  Formula: {config['formula']}")
    
    # Demonstrate class weight computation
    print("\n" + "=" * 70)
    print("Class Weight Computation Example:")
    print("-" * 70)
    
    # Simulated class distribution (imbalanced)
    class_counts = {
        0: 1000,  # Class A - many samples
        1: 500,   # Class B - medium
        2: 100,   # Class C - few samples
        3: 50     # Class D - very few samples
    }
    
    print("\nClass distribution:")
    for class_idx, count in class_counts.items():
        print(f"  Class {class_idx}: {count} samples")
    
    for method in ['balanced', 'inverse', 'sqrt_inverse']:
        weights = compute_class_weights(class_counts, method=method)
        print(f"\n{method.capitalize()} weights:")
        for class_idx, weight in weights.items():
            print(f"  Class {class_idx}: {weight:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Custom loss functions module ready!")
    print("\nNote: These are configuration templates and pseudo-implementations.")
    print("Actual implementation requires TensorFlow/PyTorch framework.")
